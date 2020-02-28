import tensorflow.keras.backend as K

# https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/master/lib/datasets/cityscapes.py
class_weights = [
    0.8373, 0.918, 0.866, 1.0345, 
    1.0166, 0.9969, 0.9754, 1.0489,
    0.8786, 1.0023, 0.9539, 0.9843, 
    1.1116, 0.9037, 1.0865, 1.0955, 
    1.0865, 1.1529, 1.0507
]

# n_lbls = 19
# s = [0] * n_lbls
# for i in range(len(gen_val)):
#     lbls = gen_val[i][1]
#     for j in range(n_lbls):
#         s[j] += lbls[..., j].sum()
# print(s)        
# for i in range(len(gen_train)):
#     lbls = gen_val[i][1]
#     for j in range(n_lbls):
#         s[j] += lbls[..., j].sum()
# print([si / 3475 for si in s])
class_frequencies = [
    345264442.0, 49568652.0, 201005428.0, 6718315.0, 
    7521741.0, 13565658.0, 1808393.0, 6098373.0, 
    158868008.0, 7625026.0, 30765347.0, 11913424.0, 
    1975596.0, 59731217.0, 2760211.0, 3563120.0, 
    1031648.0, 729415.0, 6504475.0
]
class_frequencies = [cf / 3475 for cf in class_frequencies]

def focal_loss(y_true, y_pred, gamma=2):
    # numerics
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    loss = 0
    for i in range(19):
        y_true_cls, y_pred_cls = y_true[..., i], y_pred[..., i]
        loss -= class_weights[i] * K.mean(K.sum(y_true_cls * K.pow(1 - y_pred_cls, gamma) * K.log(y_pred_cls), axis=[1, 2, 3])) / class_frequencies[i]
    loss /= 19
    return loss

def my_loss(y_true, y_pred, gamma=0):
    ''' Balances FP and FN error with unbalanced classes, which no CCE formulation I found does (didn't search too hard because it's easy). '''
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = 0
    for i in range(19):
        y_true_cls, y_pred_cls = y_true[..., i], y_pred[..., i]
        # FN
        loss -= K.mean(K.sum(y_true_cls * K.pow(1 - y_pred_cls, gamma) * K.log(y_pred_cls), axis=[1, 2])) / class_frequencies[i]
        # FP
        loss -= K.mean(K.sum((1 - y_true_cls) * K.pow(y_pred_cls, gamma) * K.log(1 - y_pred_cls), axis=[1, 2])) / class_frequencies[i]
    return loss