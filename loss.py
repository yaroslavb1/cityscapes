import numpy as np
import tensorflow.keras.backend as K

# https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/master/lib/datasets/cityscapes.py
class_weights = [
    0.8373, 0.918, 0.866, 1.0345, 
    1.0166, 0.9969, 0.9754, 1.0489,
    0.8786, 1.0023, 0.9539, 0.9843, 
    1.1116, 0.9037, 1.0865, 1.0955, 
    1.0865, 1.1529, 1.0507
]

def focal_loss(y_true, y_pred, gamma=2):
    # numerics
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    loss = 0
    for i in range(19):
        y_true_cls, y_pred_cls = y_true[..., i], y_pred[..., i]
        loss -= class_weights[i] * K.mean(y_true_cls * K.pow(1 - y_pred_cls, gamma) * K.log(y_pred_cls))
    loss /= 19
    return loss
