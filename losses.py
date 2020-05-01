from metrics import dice_coefficient, tversky
from tensorflow.keras.losses import binary_crossentropy


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coefficient(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)


def focal_tversky(y_true, y_pred, gamma=0.75):
    pt_1 = tversky(y_true, y_pred)
    return K.pow((1-pt_1), gamma)
