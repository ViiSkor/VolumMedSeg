from metrics import dice_coefficient
from tensorflow.keras.losses import binary_crossentropy


def soft_dice_loss(y_true, y_pred):
  """Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
  
    Args:
      y_true (numpy.array/tf.tensor):
        b x X x Y( x Z...) x c One hot encoding of ground truth.
      y_pred (numpy.array/tf.tensor):
        b x X x Y( x Z...) x c Network output, must sum to 1 over c channel.
    Returns:
      (tf.tensor): Soft dice loss.
  """

  loss = 1 - dice_coefficient(y_true, y_pred)
  return loss


def bce_dice_loss(y_true, y_pred):
  """Soft dice loss + binary crossentropy.
  
    Args:
      y_true (numpy.array/tf.tensor):
        b x X x Y( x Z...) x c One hot encoding of ground truth.
      y_pred (numpy.array/tf.tensor):
        b x X x Y( x Z...) x c Network output, must sum to 1 over c channel.
    Returns:
      (tf.tensor): loss.
  """

  loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
  return loss


