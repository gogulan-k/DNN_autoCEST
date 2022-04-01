import tensorflow        as tf

NP=65
FilterFactor=32
kernel=(6,1)

def build_model():
    #
    def dfh_gelu(x):
        return tf.add(tf.tanh(x), tf.math.scalar_mul( 0.02,x))
    def dfh_sigm(x):
        return tf.add(tf.math.sigmoid(x), tf.math.scalar_mul( 0.02,x))
    
    tf.keras.utils.get_custom_objects().update({'dfh_gelu': tf.keras.layers.Activation(dfh_gelu), 'dfh_sigm': tf.keras.layers.Activation(dfh_sigm)})

    time_0   =  tf.keras.layers.Input( shape=(NP*2,), name='Input_time' )
    ipap_0   =  tf.keras.layers.Input( shape=(NP*2,), name='Input_ipap' )
    
    #
    xi = tf.expand_dims( ipap_0, axis=-1 ) # ( Batch, [real,imag] )
    hi = tf.expand_dims( time_0, axis=-1 )

    def lstm_module(filters):
        def inside(x):
            # x[0]: main   track
            # x[1]: memory track

            hidden_0_1 = tf.keras.layers.Dense( filters, activation=dfh_gelu, use_bias=True)( x[1] )
            
            hidden_1_1 = tf.keras.layers.Dense( filters, activation=dfh_sigm)(x[0])
            hidden_1_2 = tf.keras.layers.Dense( filters, activation=dfh_sigm)(x[0])
            hidden_1_3 = tf.keras.layers.Dense( filters, activation=dfh_gelu)(x[0])
            hidden_1_4 = tf.keras.layers.Dense( filters, activation=dfh_sigm)(x[0])

            hidden_2_1 = tf.keras.layers.multiply([hidden_1_1,hidden_0_1])
            hidden_2_2 = tf.keras.layers.multiply([hidden_1_2,hidden_1_3])
            #
            hidden_3_1 = tf.keras.layers.add( [hidden_2_1, hidden_2_2])
            #
            hidden_4_1 = tf.keras.layers.Activation(activation=dfh_gelu)( hidden_3_1 )
            hidden_4_2 = tf.keras.layers.multiply([hidden_4_1,hidden_1_4])
            #
            return hidden_4_2, hidden_3_1
        return inside

    def conv_layer(x_shp=4, y_shp=4, kernel=11, name=''):
        def inside(x):

            x1 =  tf.expand_dims( x, axis=-2)

            x2t = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[1,1], activation=dfh_gelu, padding='valid', name='conv1d_x2t'+name )(x1)
            x2s = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[1,1], activation=dfh_sigm, padding='valid', name='conv1d_x2s'+name )(x1)
            x2  = tf.keras.layers.Conv2DTranspose( FilterFactor*2, kernel_size=kernel, dilation_rate=[1,1], padding='valid')(tf.math.multiply( x2t,x2s ))
            
            x3i = tf.keras.layers.ZeroPadding2D( ( (0,1*(kernel[0]-1)),(0,0)))(x2)
            x3t = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[2,1], activation=dfh_gelu, padding='valid', name='conv1d_x3t'+name )(x3i)
            x3s = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[2,1], activation=dfh_sigm, padding='valid', name='conv1d_x3s'+name )(x3i)
            x3  = tf.keras.layers.Conv2DTranspose( FilterFactor*2, kernel_size=kernel, dilation_rate=[1,1], padding='valid')(tf.math.multiply( x3t, x3s ))
            #
            x4i = tf.keras.layers.ZeroPadding2D( ( (0,2*(kernel[0]-1)),(0,0)))(x3)
            x4t = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[3,1], activation=dfh_gelu, padding='valid', name='conv1d_x4t'+name )(x4i)
            x4s = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[3,1], activation=dfh_sigm, padding='valid', name='conv1d_x4s'+name )(x4i)
            x4  = tf.keras.layers.Conv2DTranspose( FilterFactor*2, kernel_size=kernel, dilation_rate=[1,1], padding='valid')(tf.math.multiply( x4t, x4s ))

            x5i = tf.keras.layers.ZeroPadding2D( ( (0,3*(kernel[0]-1)),(0,0)))(x4)
            x5t = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[4,1], activation=dfh_gelu, padding='valid', name='conv1d_x5t'+name )(x5i)
            x5s = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[4,1], activation=dfh_sigm, padding='valid', name='conv1d_x5s'+name )(x5i)
            x5  = tf.keras.layers.Conv2DTranspose( FilterFactor*2, kernel_size=kernel, dilation_rate=[1,1], padding='valid')(tf.math.multiply(x5t,x5s))

            x6i = tf.keras.layers.ZeroPadding2D( ( (0,5*(kernel[0]-1)),(0,0)))(x5)
            x6t = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[6,1],activation=dfh_gelu, padding='valid', name='conv1d_x6t'+name )(x6i)
            x6s = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[6,1],activation=dfh_sigm, padding='valid', name='conv1d_x6s'+name )(x6i)
            x6  = tf.keras.layers.Conv2DTranspose( FilterFactor*2, kernel_size=kernel, dilation_rate=[1,1], padding='valid')(tf.math.multiply(x6t,x6s))

            x7i = tf.keras.layers.ZeroPadding2D( ( (0,7*(kernel[0]-1)),(0,0)))(x6)
            x7t = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[8,1],activation=dfh_gelu, padding='valid', name='conv1d_x7t'+name )(x7i)
            x7s = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[8,1],activation=dfh_sigm, padding='valid', name='conv1d_x7s'+name )(x7i)
            x7  = tf.keras.layers.Conv2DTranspose( FilterFactor*2, kernel_size=kernel, dilation_rate=[1,1], padding='valid')(tf.math.multiply(x7t,x7s))
            #
            x8i = tf.keras.layers.ZeroPadding2D( ( (0,9*(kernel[0]-1)),(0,0)))(x7)
            x8t = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[10,1],activation=dfh_gelu, padding='valid', name='conv1d_x8t'+name )(x8i)
            x8s = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[10,1],activation=dfh_sigm, padding='valid', name='conv1d_x8s'+name )(x8i)
            x8  = tf.keras.layers.Conv2DTranspose( FilterFactor*2, kernel_size=kernel, dilation_rate=[1,1], padding='valid')(tf.math.multiply(x8t,x8s))

            x9i = tf.keras.layers.ZeroPadding2D( ( (0,11*(kernel[0]-1)),(0,0)))(x8)
            x9t = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[12,1],activation=dfh_gelu, padding='valid', name='conv1d_x9t'+name )(x9i)
            x9s = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[12,1],activation=dfh_sigm, padding='valid', name='conv1d_x9s'+name )(x9i)
            x9  = tf.keras.layers.Conv2DTranspose( FilterFactor*2, kernel_size=kernel, dilation_rate=[1,1], padding='valid')(tf.math.multiply(x9t,x9s))

            x10i = tf.keras.layers.ZeroPadding2D( ( (0,13*(kernel[0]-1)),(0,0)))(x9)
            x10t = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[14,1],activation=dfh_gelu, padding='valid', name='conv1d_x10t'+name )(x10i)
            x10s = tf.keras.layers.Conv2D( FilterFactor, kernel_size=kernel, dilation_rate=[14,1],activation=dfh_sigm, padding='valid', name='conv1d_x10s'+name )(x10i)
            x10  = tf.keras.layers.Conv2DTranspose( FilterFactor*2, kernel_size=kernel, dilation_rate=[1,1], padding='valid')(tf.math.multiply(x10t,x10s))

            x15 = tf.keras.layers.Add()([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10])

            return tf.squeeze( x15, axis=2)
            
        return inside
    
    def lstm_layer(x_shp=4, y_shp=4, kernel=11, name=''):
        def inside(x):

            x0 = conv_layer(x_shp=x_shp, y_shp=y_shp, kernel=kernel, name='x_'+name)(x[0])
            h0 = conv_layer(x_shp=x_shp, y_shp=y_shp, kernel=kernel, name='h_'+name)(x[1])
            #
            x00 = tf.keras.layers.Permute( (2,1))(x0)
            h00 = tf.keras.layers.Permute( (2,1))(h0)
            #
            x11, h11 = lstm_module( x_shp )( [ x00, h00] )
            #
            x1 = tf.keras.layers.Permute( (2,1))(x11)
            h1 = tf.keras.layers.Permute( (2,1))(h11)
            #
            # final transformation
            x0p = tf.keras.layers.add( [ x0, \
                                         tf.keras.layers.multiply( [ \
                                                        tf.keras.layers.Dense( FilterFactor*2, activation=dfh_gelu)(x[1]), \
                                                        tf.keras.layers.Dense( FilterFactor*2, activation=dfh_gelu)(x1)]) ] )
            #
            h0p = tf.keras.layers.add( [ h0, \
                                         tf.keras.layers.multiply( [ \
                                                        tf.keras.layers.Dense( FilterFactor*2, activation=dfh_gelu)(x[0]), \
                                                        tf.keras.layers.Dense( FilterFactor*2, activation=dfh_gelu)(h1)]) ] )

            return x0p,h0p
        return inside

    x1,h1 = lstm_layer(x_shp=2*NP,y_shp=1, kernel=kernel, name='_l1')([ xi,hi] )
    x2,h2 = lstm_layer(x_shp=2*NP,y_shp=1, kernel=kernel, name='_l2')([ x1,h1] )
    x3,h3 = lstm_layer(x_shp=2*NP,y_shp=1, kernel=kernel, name='_l3')([ x2,h2] )
    x4,h4 = lstm_layer(x_shp=2*NP,y_shp=1, kernel=kernel, name='_l4')([ x3,h3] )
    x5,h5 = lstm_layer(x_shp=2*NP,y_shp=1, kernel=kernel, name='_l5')([ x4,h4] )
    #
    xhs = tf.keras.layers.concatenate([x1,x2,x3,x4,x5,h1,h2,h3,h4,h5], axis=2)
    xhf = tf.keras.layers.Dense( 1, activation=dfh_gelu)( xhs )
    #    
    final_rshp  = tf.squeeze( tf.keras.layers.Activation('linear', dtype='float32')( xhf ), axis=-1 )
    final_rshp  = tf.math.scalar_mul( tf.constant( NP, dtype=tf.dtypes.float32 ), final_rshp )
    #    
    final = tf.keras.models.Model(inputs=(time_0, ipap_0), outputs=final_rshp)
    #
    final.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['mse','mae'])
    return final


