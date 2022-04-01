    
def build_model_denseConv(filters=16):
    #
    import tensorflow as     tf
    #

    def dense_block(x, filty, num_layers):
        stack = []
        x = alt_conv(x, filty)
        stack.append(x)

        for k in range(num_layers-1):
            if k>0:
                x = alt_conv(keras.layers.Concatenate()(stack), filty)
            else:
                x = alt_conv(x, filty)

            stack.append(x)

        return keras.layers.Concatenate()(stack)

    def alt_conv(x, filty):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv1D(4*filty, kernel_size=1, strides=1, padding = 'same')(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv1D(filty, kernel_size=3, strides=1, padding = 'same')(x)
        return x

    def transition(x):
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.AveragePooling1D(pool_size=2, strides = 2)(x)

        return x

    input =  keras.layers.Input(shape=[128,1])
    x = input
    x = keras.layers.Conv1D(filters, kernel_size=7, strides=1, padding = 'same')(x)
    x = keras.layers.MaxPool1D(pool_size=3, strides=2)(x)

    x = dense_block(x, filters, 6)
    x = transition(x)

    x = dense_block(x, filters, 12)
    x = transition(x)

    x = dense_block(x, filters, 32)
    x = transition(x)

    x = dense_block(x, filters, 32)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    fin_dens = keras.layers.Dense(6, activation="sigmoid")(x)
    fin_dens_reshape = keras.layers.Reshape((3, 2), input_shape=(6,))(fin_dens)
    #
    model = keras.Model(inputs=[input], outputs=[fin_dens_reshape])
    #
    model.compile(loss=unique_pairs_loss,
                  optimizer=keras.optimizers.Adam(lr=0.33e-4),
                  metrics=[unique_pairs_loss, conf_loss]
    )
    #
    return model
