from tensorflow.keras.layers import Conv2D, Conv3D, BatchNormalization, Activation, SpatialDropout2D, SpatialDropout3D, Dropout


def conv_block(inputs, n_filters, conv_kwds, activation, dropout_prob, conv_type="3D", dropout_type=None, batchnorm=False):
    if conv_type == "2D":
      conv = Conv2D
      spatial_dropout = SpatialDropout2D
    elif conv_type == "3D":
      conv = Conv3D
      spatial_dropout = SpatialDropout3D
    else:
      raise ValueError(f"conv_type must be one of ['2D', '3D'], but got {conv_type}")
    
    if dropout_type == "standard":
        dropout = Dropout
    elif dropout_type == "spatial":
        dropout = spatial_dropout
    else:
      if dropout_type:
        raise ValueError(f"dropout_type must be one of ['standard', 'spatial', None], but got {dropout_type}")

    # first layer
    x = conv(filters=n_filters, **conv_kwds)(inputs)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if dropout_type and dropout_prob > 0.0:
        x = dropout(dropout_prob)(x)

    # second layer
    x = conv(filters=n_filters, **conv_kwds)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x
