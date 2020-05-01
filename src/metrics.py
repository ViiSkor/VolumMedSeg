from tensorflow.keras import backend as K

smooth = 1.

def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def confusion(y_true, y_pred):
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg) 
    prec = (tp + smooth)/(tp+fp+smooth)
    rec = (tp+smooth)/(tp+fn+smooth)
    return prec, rec


def recall(y_true, y_pred):
  y_pred_pos = K.clip(y_pred, 0, 1)
  y_pred_neg = 1 - y_pred_pos
  y_pos = K.clip(y_true, 0, 1)
  y_neg = 1 - y_pos
  tp = K.sum(y_pos * y_pred_pos)
  fp = K.sum(y_neg * y_pred_pos)
  fn = K.sum(y_pos * y_pred_neg)
  rec = (tp+smooth)/(tp+fn+smooth)
  return rec


def precision(y_true, y_pred):
  y_pred_pos = K.clip(y_pred, 0, 1)
  y_pred_neg = 1 - y_pred_pos
  y_pos = K.clip(y_true, 0, 1)
  y_neg = 1 - y_pos
  tp = K.sum(y_pos * y_pred_pos)
  fp = K.sum(y_neg * y_pred_pos)
  fn = K.sum(y_pos * y_pred_neg)
  prec = (tp + smooth)/(tp+fp+smooth)
  return prec


def tversky(y_true, y_pred, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
