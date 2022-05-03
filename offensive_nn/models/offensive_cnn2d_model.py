import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class OffensiveCNN2DModel:
    def __init__(self, args, embedding_matrix=None):
        filter_sizes = [1, 2, 3, 5]
        num_filters = 32

        emb = layers.Embedding(args.max_features, args.embed_size, trainable=False,
                               name="embedding_layer")

        inp = tf.keras.Input(shape=(None,), dtype="int64", name="input")
        x = emb(inp)
        x = layers.SpatialDropout1D(0.4, name="spatial_dropout_layer")(x)
        x = layers.Reshape((args.max_len, args.embed_size, 1), name="reshape_layer")(x)

        conv_0 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[0], args.embed_size), kernel_initializer='normal',
                        activation='elu', name="conv0_layer")(x)
        conv_1 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[1], args.embed_size), kernel_initializer='normal',
                        activation='elu', name="conv1_layer")(x)
        conv_2 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[2], args.embed_size), kernel_initializer='normal',
                        activation='elu', name="conv2_layer")(x)
        conv_3 = layers.Conv2D(num_filters, kernel_size=(filter_sizes[3], args.embed_size), kernel_initializer='normal',
                        activation='elu', name="conv3_layer")(x)

        maxpool_0 = layers.MaxPool2D(pool_size=(args.max_len - filter_sizes[0] + 1, 1), name="pool0_layer")(conv_0)
        maxpool_1 = layers.MaxPool2D(pool_size=(args.max_len - filter_sizes[1] + 1, 1), name="pool1_layer")(conv_1)
        maxpool_2 = layers.MaxPool2D(pool_size=(args.max_len - filter_sizes[2] + 1, 1), name="pool2_layer")(conv_2)
        maxpool_3 = layers.MaxPool2D(pool_size=(args.max_len - filter_sizes[3] + 1, 1), name="pool3_layer")(conv_3)

        z = layers.Concatenate(axis=1, name="conc_layer")([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
        z = layers.Flatten(name="flatten_layer")(z)
        z = layers.Dropout(0.1, name="dropout_layer")(z)
        outp = layers.Dense(args.num_classes, activation="softmax", name="dense_predictions")(z)
        emb.set_weights([embedding_matrix])
        self.model = tf.keras.Model(inputs=inp, outputs=outp, name="cnn_model")
