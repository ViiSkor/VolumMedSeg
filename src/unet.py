from tensorflow.keras import Model
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Input, BatchNormalization, Activation, MaxPool3D, SpatialDropout3D, Concatenate


class Unet3D:
  def __init__(self,
               n_classes,
               input_shape,
               activation="relu",
               n_base_filters=8,
               batchnorm=False,
               spatial_dropout=False,
               batch_size=None,
               model_depth=5,
               name="3DUnet"):
    self.n_classes = n_classes
    self.input_shape = input_shape
    self.activation = activation
    self.n_base_filters = n_base_filters
    self.batchnorm = batchnorm
    self.spatial_dropout = spatial_dropout
    self.batch_size = batch_size
    self.model_depth = model_depth
    self.name = name
    
    self.skips = []

    self.conv_kwds = {
          "kernel_size": (3, 3, 3),
          "activation": None,
          "padding": "same",
          "kernel_initializer": "he_normal",
          # 'kernel_regularizer': tf.keras.regularizers.l2(0.001),
      }

    self.conv_transpose_kwds = {
          "kernel_size": (2, 2, 2),
          "strides": 2,
          "padding": "same",
          "kernel_initializer": "he_normal",
          # 'kernel_regularizer': tf.keras.regularizers.l2(0.001),
      }

  def encoder(self, inputs):
    x = inputs
    for depth in range(self.model_depth):
      x = Conv3D(self.n_base_filters * (2**depth), **self.conv_kwds)(x)
      if self.batchnorm:
          x = BatchNormalization()(x)
      x = Activation(self.activation)(x)
      x = Conv3D(self.n_base_filters * (2**(depth+1)), **self.conv_kwds)(x)
      if self.batchnorm:
          x = BatchNormalization()(x)
      x = Activation(self.activation)(x)
      if depth < self.model_depth - 1:
        self.skips.append(x)
        x = MaxPool3D(2)(x)
        if self.spatial_dropout:
          x = SpatialDropout3D(0.5)(x)

    return x

  def decoder(self, x):
    for depth in range(self.model_depth-1, 0, -1):
      x = Conv3DTranspose(self.n_base_filters * (2**(depth+1)), **self.conv_transpose_kwds)(x)

      x = Concatenate(axis=-1)([self.skips[depth-1], x])
      if self.spatial_dropout:
          x = SpatialDropout3D(0.5)(x)
      x = Conv3D(self.n_base_filters * (2**depth), **self.conv_kwds)(x)
      if self.batchnorm:
          x = BatchNormalization()(x)
      x = Activation(self.activation)(x)
      x = Conv3D(self.n_base_filters * (2**depth), **self.conv_kwds)(x)
      if self.batchnorm:
          x = BatchNormalization()(x)
      x = Activation(self.activation)(x)

    x = Conv3D(filters=self.n_classes, kernel_size=1)(x)
    return x


  def build_model(self):
    inputs = Input(shape=self.input_shape, batch_size=self.batch_size)
    x = self.encoder(inputs)
    x = self.decoder(x)

    final_activation = "sigmoid" if self.n_classes == 1 else "softmax"
    x = Activation(final_activation)(x)

    model = Model(inputs=inputs, outputs=x, name=self.name)
    return model
