import io
import logging
import os
import random
import shutil

import absl.logging
import gensim.downloader as api
import gensim.utils
import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from offensive_nn.model_args import ModelArgs
from offensive_nn.models.offensive_cnn1d_model import OffensiveCNN1DModel
from offensive_nn.models.offensive_cnn2d_model import OffensiveCNN2DModel
from offensive_nn.models.offensive_lstm_model import OffensiveLSTMModel

logging.basicConfig()
logging.root.setLevel(logging.INFO)

absl.logging.set_verbosity(absl.logging.ERROR)

logger = logging.getLogger(__name__)


class OffensiveNNModel:
    def __init__(self, model_type_or_path,
                 embedding_model_name_or_path=None,
                 train_df=None,
                 eval_df=None,
                 args=None,
                 emd_file=None):

        if os.path.isdir(model_type_or_path):
            self.model = keras.models.load_model(model_type_or_path)
            self.args = self._load_model_args(model_type_or_path)

        else:
            self.train_df = train_df
            self.eval_df = eval_df

            self.args = self._load_model_args(model_type_or_path)

            if isinstance(args, dict):
                self.args.update_from_dict(args)
            elif isinstance(args, ModelArgs):
                self.args = args

            if self.args.manual_seed:
                random.seed(self.args.manual_seed)
                np.random.seed(self.args.manual_seed)
                tf.random.set_seed(self.args.manual_seed)

            self.train_texts, self.train_labels = self._prepare_data(self.train_df)
            self.eval_texts, self.eval_labels = self._prepare_data(self.eval_df)

            self.vectorizer = tf.keras.layers.TextVectorization(max_tokens=self.args.max_features,
                                                                output_sequence_length=self.args.max_len)
            self.train_ds = tf.data.Dataset.from_tensor_slices(self.train_texts).batch(self.args.train_batch_size)
            self.vectorizer.adapt(self.train_ds)

            voc = self.vectorizer.get_vocabulary()
            self.word_index = dict(zip(voc, range(len(voc))))

            self.args.max_features = len(self.word_index) + 1

            # if transfer learning enabled, load the embedding matrix
            if emd_file is not None:
                self.transfer_learning = True
                transfer_learn_wv = self.load_from_vec_file(emd_file=emd_file)
                self.transfer_learning_embedding_matrix = self.get_emb_matrix(self.word_index,
                                                                              self.args.max_features,
                                                                              transfer_learn_wv)
            else:
                self.transfer_learning = False

            # load embeddings from file,api or no embeddings if embedding_model_name_or_path is None
            if embedding_model_name_or_path is not None:
                if os.path.isfile(embedding_model_name_or_path):
                    save_load = gensim.utils.SaveLoad()
                    obj = save_load.load(fname=embedding_model_name_or_path)
                    self.embedding_model = obj.wv
                else:
                    self.embedding_model = api.load(embedding_model_name_or_path)

                self.primary_embedding_matrix = self.get_emb_matrix(self.word_index, self.args.max_features,
                                                                    self.embedding_model)
                if self.transfer_learning:
                    self.embedding_matrix = np.append(self.transfer_learning_embedding_matrix,
                                                      self.primary_embedding_matrix, axis=0)
                    self.args.max_features = (len(self.word_index) + 1) * 2
                else:
                    self.embedding_matrix = self.primary_embedding_matrix
            else:
                self.embedding_matrix = None

            MODEL_CLASSES = {
                "cnn1D": OffensiveCNN1DModel,
                "cnn2D": OffensiveCNN2DModel,
                "lstm": OffensiveLSTMModel
            }

            self.model = MODEL_CLASSES[model_type_or_path](self.args, self.embedding_matrix).model

            opt = keras.optimizers.Adam(learning_rate=self.args.learning_rate)
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        logger.info(self.model.summary())

    # only to be used when transfer learning
    def transfer_learn_train_model(self,
                    args=None,
                    verbose=1, train_df=None, eval_df=None):
        if train_df is not None:
            self.train_df = train_df
            self.train_texts, self.train_labels = self._prepare_data(self.train_df)
        if eval_df is not None:
            self.eval_df = eval_df
            self.eval_texts, self.eval_labels = self._prepare_data(self.eval_df)
        self.train_model(args, verbose)

    def train_model(self,
                    args=None,
                    verbose=1):
        if os.path.exists(self.args.cache_dir) and os.path.isdir(self.args.cache_dir):
            shutil.rmtree(self.args.cache_dir)

        if os.path.exists(self.args.best_model_dir) and os.path.isdir(self.args.best_model_dir):
            shutil.rmtree(self.args.best_model_dir)

        if args:
            self.args.update_from_dict(args)
        callbacks = []

        if self.args.save_best_model:
            checkpoint = ModelCheckpoint(self.args.cache_dir, monitor='val_loss', verbose=verbose, save_best_only=True,
                                         mode='min')
            callbacks.append(checkpoint)

        if self.args.reduce_lr_on_plateau:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.args.reduce_lr_on_plateau_factor,
                                          patience=self.args.reduce_lr_on_plateau_patience,
                                          min_lr=self.args.reduce_lr_on_plateau_min_lr, verbose=1)
            callbacks.append(reduce_lr)

        if self.args.early_stopping:
            earlystopping = EarlyStopping(monitor='val_loss', min_delta=self.args.early_stopping_min_delta,
                                          patience=self.args.early_stopping_patience, verbose=verbose, mode='auto')
            callbacks.append(earlystopping)

        x_train = self.vectorizer(np.array([[s] for s in self.train_texts])).numpy()
        x_val = self.vectorizer(np.array([[s] for s in self.eval_texts])).numpy()

        y_train = np.array(self.train_labels)
        y_val = np.array(self.eval_labels)

        self.model.fit(x_train, y_train, batch_size=self.args.train_batch_size,
                       epochs=self.args.num_train_epochs, validation_data=(x_val, y_val),
                       verbose=verbose, callbacks=callbacks)

        self.model = keras.models.load_model(self.args.cache_dir)
        self.save_model()

    def predict(self, texts):
        predictions = self.model.predict(texts, batch_size=self.args.test_batch_size)
        return np.argmax(predictions, axis=1).tolist(), predictions.tolist()

    def save_model(self):
        os.makedirs(self.args.best_model_dir, exist_ok=True)

        inputs = tf.keras.Input(shape=(1,), dtype="string")
        # Turn strings into vocab indices
        indices = self.vectorizer(inputs)
        # Turn vocab indices into predictions
        outputs = self.model(indices)

        end_to_end_model = tf.keras.Model(inputs, outputs)
        end_to_end_model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        end_to_end_model.save(self.args.best_model_dir)
        self.save_model_args(self.args.best_model_dir)

    @staticmethod
    def _prepare_data(data_frame):
        texts = []
        labels = []
        for index, row in data_frame.iterrows():
            texts.append(row['text'])
            labels.append(row['labels'])

        return texts, labels

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    @staticmethod
    def _load_model_args(input_dir):
        args = ModelArgs()
        args.load(input_dir)
        return args

    @staticmethod
    def load_word_emb(word_index, emebdding_model):
        embeddings_index = dict()
        for idx, key in enumerate(emebdding_model.key_to_index):
            if key in word_index:
                embeddings_index[key] = emebdding_model[key]
        return embeddings_index

    def get_emb_matrix(self, word_index, max_features, embedding_file):
        embeddings_index = self.load_word_emb(word_index, embedding_file)
        all_embs = np.stack(list(embeddings_index.values()))
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
        embedding_count = 0
        no_embedding_count = 0
        for word, i in word_index.items():
            if i >= max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                embedding_count = embedding_count + 1
            else:
                no_embedding_count = no_embedding_count + 1

        no_embedding_rate = no_embedding_count / (embedding_count + no_embedding_count)
        logger.warning("Embeddings are not found for {:.2f}% words.".format(no_embedding_rate * 100))

        return embedding_matrix

    @staticmethod
    def load_from_vec_file(emd_file):
        w2v_model = KeyedVectors.load_word2vec_format(emd_file)
        return w2v_model
