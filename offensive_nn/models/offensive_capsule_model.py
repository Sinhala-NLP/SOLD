from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Embedding, SpatialDropout1D, Bidirectional, GRU, Flatten, Dense, Dropout, \
    BatchNormalization
from tensorflow.python.ops.init_ops_v2 import glorot_normal
from tensorflow.python.ops.initializers_ns import orthogonal

from offensive_nn.models.layers import Capsule


class OffensiveCapsuleModel:
    def __init__(self, args, embedding_matrix):
        inp = Input(shape=(args.max_len,))

        x = Embedding(args.max_features, args.embed_size, weights=[embedding_matrix], trainable=False)(inp)

        x = SpatialDropout1D(rate=0.2)(x)
        x = Bidirectional(GRU(100, return_sequences=True,
                              kernel_initializer=glorot_normal(seed=12300),
                              recurrent_initializer=orthogonal(gain=1.0, seed=10000)))(x)

        x = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x)
        x = Flatten()(x)

        x = Dense(100, activation="relu", kernel_initializer=glorot_normal(seed=12300))(x)
        x = Dropout(0.12)(x)
        x = BatchNormalization()(x)

        x = Dense(args.num_classes, activation="sigmoid")(x)
        self.model = Model(inputs=inp, outputs=x)
        # model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
