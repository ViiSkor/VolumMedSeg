from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, Input, Activation, MaxPool2D, MaxPool3D, Concatenate
from tensorflow.keras.regularizers import l2

from blocks import conv_block, dilate_conv_block


class Unet:
  """
  The class of U-Net architecture [1].

  Attributes:
  n_classes (int): Unique classes in the output mask.
  input_shape: Tensor of shape [x, y, channels]/[x, y, z, channels]
  activation (str): A tensorflow.keras.activations.Activation to use.
  n_base_filters (int): Convolutional filters in the initial convolutional block. Will be doubled every block.
  batchnorm (bool): Use Batch Normalisation or not
  dropout_prob (float): The probobility to eluminate a nerone after the initial convolutional block. Set to 0. to turn Dropout off
  dropout_type (one of "spatial" or "standard"): Type of Dropout to apply.
  dropout_prob_shift (float between 0. and 1.): Factor to add to the Dropout after each conv block.
  batch_size (int): The subset size of a training sample.
  model_depth (int): The number of blocks in decoder and encoder.
  dilate (bool): Set to True to use dilated convolution.
  bottleneck_depth (int): Number of layers in the bottleneck. Not matter if  dilate is True.
  max_dilation_rate (int): Num of holes in the last conv layer in the bottleneck. Will set the number of layers in the bottleneck equal to ceil(log2(max_dilation_rate)-1.
  name (str): Name of assembled model.
  mode (one of "2D" or "3D"): Set type of U-Net.
  Returns:
  model (tensorflow.keras.models.Model): The built U-Net.
    
  [1]: https://arxiv.org/abs/1505.04597
  """

  def __init__(self,
               n_classes,
               input_shape,
               activation="relu",
               n_base_filters=64,
               batchnorm=False,
               dropout_prob=0.2,
               dropout_type="spatial",
               dropout_prob_shift=0.1,
               batch_size=None,
               model_depth=4,
               dilate=False,
               bottleneck_depth=2,
               max_dilation_rate=32,
               name="Unet",
               mode="3D"):
    self.n_classes = n_classes
    self.input_shape = input_shape
    self.activation = activation
    self.n_base_filters = n_base_filters
    self.batchnorm = batchnorm
    self.dropout_prob = dropout_prob
    self.dropout_type = dropout_type
    self.dropout_prob_shift = dropout_prob_shift
    self.batch_size = batch_size
    self.model_depth = model_depth
    self.bottleneck_depth = bottleneck_depth
    self.dilate = dilate
    self.max_dilation_rate = max_dilation_rate
    self.name = name
    self.mode = mode

    self.skips = []
    self.__set_layers()
    self.__set_layers_prms()

  def __set_layers(self):
    if self.mode == "2D":
      self.conv = Conv2D
      self.transpose = Conv2DTranspose
      self.maxpool = MaxPool2D
    elif self.mode == "3D":
      self.conv = Conv3D
      self.transpose = Conv3DTranspose
      self.maxpool = MaxPool3D
    else:
      raise ValueError(f"'mode' must be one of ['2D', '3D'], but got {self.mode}")

  def __set_layers_prms(self):
    if self.mode == "2D":
      self.conv_kwds = {
          "kernel_size": (3, 3),
          "activation": None,
          "padding": "same",
          "kernel_initializer": "he_normal",
          "kernel_regularizer": l2(0.001)
      }

      self.conv_transpose_kwds = {
            "kernel_size": (2, 2),
            "strides": 2,
            "padding": "same",
            "kernel_initializer": "he_normal",
            "kernel_regularizer": l2(0.001)
        }
    elif self.mode == "3D":
      self.conv_kwds = {
          "kernel_size": (3, 3, 3),
          "activation": None,
          "padding": "same",
          "kernel_initializer": "he_normal",
          "kernel_regularizer": l2(0.001)
      }

      self.conv_transpose_kwds = {
            "kernel_size": (2, 2, 2),
            "strides": 2,
            "padding": "same",
            "kernel_initializer": "he_normal",
            "kernel_regularizer": l2(0.001)
        }
    else:
      raise ValueError(f"'mode' must be one of ['2D', '3D'], but got {self.mode}")
    

  def encoder(self, inputs):
    x = inputs
    for depth in range(self.model_depth):
      filters = self.n_base_filters * (2**depth)
      x = conv_block(inputs=x,
                     n_filters=filters,
                     conv_kwds=self.conv_kwds,
                     activation=self.activation,
                     dropout_prob=self.dropout_prob,
                     conv_type=self.mode,
                     dropout_type=self.dropout_type,
                     batchnorm=self.batchnorm)
      if depth < self.model_depth:
        self.skips.append(x)
        x = self.maxpool(2)(x)

      self.dropout_prob += self.dropout_prob_shift

    return x

  def bottleneck(self, x):
    filters = self.n_base_filters * (2**self.model_depth)
    if self.dilate:
      x = dilate_conv_block(x=x,
                   n_filters=filters,
                   max_dilation_rate=self.max_dilation_rate,
                   conv_kwds=self.conv_kwds,
                   activation=self.activation,
                   dropout_prob=self.dropout_prob,
                   conv_type=self.mode,
                   dropout_type=self.dropout_type,
                   batchnorm=self.batchnorm)
    else:
      for _ in range(1, self.bottleneck_depth, 2):
        x = conv_block(inputs=x,
                   n_filters=filters,
                   conv_kwds=self.conv_kwds,
                   activation=self.activation,
                   dropout_prob=self.dropout_prob,
                   conv_type=self.mode,
                   dropout_type=self.dropout_type,
                   batchnorm=self.batchnorm)
        x = conv_block(inputs=x,
                   n_filters=filters,
                   conv_kwds=self.conv_kwds,
                   activation=self.activation,
                   dropout_prob=self.dropout_prob,
                   conv_type=self.mode,
                   dropout_type=self.dropout_type,
                   batchnorm=self.batchnorm)
      if self.bottleneck_depth % 2 != 0:
        x = self.conv(filters=filters, **self.conv_kwds)(x)

    return x

  def decoder(self, x):
    for depth in range(self.model_depth, 0, -1):
      filters_upsampling = self.n_base_filters * (2**depth)
      filters_conv = self.n_base_filters * (2**(depth-1))
      self.dropout_prob -= self.dropout_prob_shift

      x = self.transpose(filters_upsampling, **self.conv_transpose_kwds)(x)
      x = Concatenate(axis=-1)([self.skips[depth-1], x])
      x = conv_block(inputs=x,
                     n_filters=filters_conv,
                     conv_kwds=self.conv_kwds,
                     activation=self.activation,
                     dropout_prob=self.dropout_prob,
                     conv_type=self.mode,
                     dropout_type=self.dropout_type,
                     batchnorm=self.batchnorm)

    x = self.conv(filters=self.n_classes, kernel_size=1)(x)
    return x

  def build_model(self):
    inputs = Input(shape=self.input_shape, batch_size=self.batch_size)
    x = self.encoder(inputs)
    x = self.bottleneck(x)
    x = self.decoder(x)

    final_activation = "sigmoid" if self.n_classes == 1 else "softmax"
    x = Activation(final_activation)(x)

    model = Model(inputs=inputs, outputs=x, name=self.name)
    return model
