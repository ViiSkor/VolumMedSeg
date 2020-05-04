from tensorflow.keras.layers import Conv3D, BatchNormalization, Activation, SpatialDropout3D, Dropout

def conv3d_block(inputs, n_filters, conv_kwds, activation, dropout_prob, dropout_type=None, batchnorm=False):
    if dropout_type == "standard":
        dropout = Dropout
    elif dropout_type == "spatial":
        dropout = SpatialDropout3D

    # first layer
    x = Conv3D(filters=n_filters, **conv_kwds)(inputs)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if dropout_type and dropout_prob > 0.0:
        x = dropout(dropout_prob)(x)

    # second layer
    x = Conv3D(filters=n_filters, **conv_kwds)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x
