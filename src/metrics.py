import numpy as np
from tensorflow.keras import backend as K

epsilon=1e-6 # Used for numerical stability to avoid divide by zero errors


def dice_coefficient(y_true, y_pred): 
    """Dice coefficient calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
  
    Args:
      y_true (numpy.array/tf.tensor):
        b x X x Y( x Z...) x c One hot encoding of ground truth.
      y_pred (numpy.array/tf.tensor):
        b x X x Y( x Z...) x c Network output, must sum to 1 over c channel.
    Returns:
      (tf.tensor): Dice coefficient.
    References
      Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    
    return K.mean(numerator / (denominator + epsilon)) # average over classes and batch


def confusion(y_true, y_pred):
  """Confusion matrix.
  
  Args:
  y_true (numpy.array/tf.tensor):
    b x X x Y( x Z...) x c One hot encoding of ground truth.
  y_pred (numpy.array/tf.tensor):
    b x X x Y( x Z...) x c Network output, must sum to 1 over c channel.
  Returns:
    (tf.tensor): precision, recall.
  """

  y_pred_pos = K.clip(y_pred, 0, 1)
  y_pred_neg = 1 - y_pred_pos
  y_pos = K.clip(y_true, 0, 1)
  y_neg = 1 - y_pos
  tp = K.sum(y_pos * y_pred_pos)
  fp = K.sum(y_neg * y_pred_pos)
  fn = K.sum(y_pos * y_pred_neg) 
  prec = (tp)/(tp+fp+epsilon)
  rec = (tp)/(tp+fn+epsilon)
  return prec, rec


def recall(y_true, y_pred):
  """Recall=True_positive/(True_positive + False_negative).
  
  # Args:
    y_true (numpy.array/tf.tensor):
      b x X x Y( x Z...) x c One hot encoding of ground truth.
    y_pred (numpy.array/tf.tensor):
      b x X x Y( x Z...) x c Network output, must sum to 1 over c channel.
  Returns:
    (tf.tensor): recall.
  """

  y_pred_pos = K.clip(y_pred, 0, 1)
  y_pred_neg = 1 - y_pred_pos
  y_pos = K.clip(y_true, 0, 1)
  y_neg = 1 - y_pos
  tp = K.sum(y_pos * y_pred_pos)
  fp = K.sum(y_neg * y_pred_pos)
  fn = K.sum(y_pos * y_pred_neg)
  rec = (tp)/(tp+fn+epsilon)
  return rec


def precision(y_true, y_pred):
  """Precision=True_positive/(True_positive + False_positive).
  
  # Args:
    y_true (numpy.array/tf.tensor):
      b x X x Y( x Z...) x c One hot encoding of ground truth.
    y_pred (numpy.array/tf.tensor):
      b x X x Y( x Z...) x c Network output, must sum to 1 over c channel.
  Returns:
    (tf.tensor): precision.
  """

  y_pred_pos = K.clip(y_pred, 0, 1)
  y_pred_neg = 1 - y_pred_pos
  y_pos = K.clip(y_true, 0, 1)
  y_neg = 1 - y_pos
  tp = K.sum(y_pos * y_pred_pos)
  fp = K.sum(y_neg * y_pred_pos)
  fn = K.sum(y_pos * y_pred_neg)
  prec = (tp)/(tp+fp+epsilon)
  return prec


def iou(y_true, y_pred): 
    """Intersection over Union for arbitrary batch size, number of classes, and number of spatial dimensions.
  
    Args:
      y_true (numpy.array/tf.tensor):
        b x X x Y( x Z...) x c One hot encoding of ground truth.
      y_pred (numpy.array/tf.tensor):
        b x X x Y( x Z...) x c Network output, must sum to 1 over c channel.
    Returns:
      (tf.tensor): IoU coefficient.
    """

    # skip the batch and class axis for calculating IoU score
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    return K.mean(numerator / (denominator + epsilon)) # average over classes and batch


def mean_iou(y_true, y_pred):
    """Mean of Intersection over Union.
    Thresholds for mask are from 0.5, to 1.0 with a step 0.5.
  
    Args:
      y_true (numpy.array/tf.tensor):
        b x X x Y( x Z...) x c One hot encoding of ground truth.
      y_pred (numpy.array/tf.tensor):
        b x X x Y( x Z...) x c Network output, must sum to 1 over c channel.
    Returns:
      (tf.tensor): Mean IoU coefficient.
    """

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
      y_pred = K.cast(K.greater(y_pred, t), dtype='float32')
      score = iou(y_true, y_pred)
      prec.append(score)


def mean_dice(y_true, y_pred):
    """Mean of dice score.
    Thresholds for mask are from 0.5, to 1.0 with a step 0.5.
  
    Args:
      y_true (numpy.array/tf.tensor):
        b x X x Y( x Z...) x c One hot encoding of ground truth.
      y_pred (numpy.array/tf.tensor):
        b x X x Y( x Z...) x c Network output, must sum to 1 over c channel.
    Returns:
      (tf.tensor): Mean IoU coefficient.
    """

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
      y_pred = K.cast(K.greater(y_pred, t), dtype='float32')
      y_pred = K.constant(y_pred)
      y_true = K.constant(y_true)
      score = dice_coefficient(y_true, y_pred)
      prec.append(score)

    return K.mean(K.stack(prec), axis=0)
