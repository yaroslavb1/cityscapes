import tensorflow.keras.backend as K

# https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/master/lib/datasets/cityscapes.py
class_weights = [
    0.8373, 0.918, 0.866, 1.0345, 
    1.0166, 0.9969, 0.9754, 1.0489,
    0.8786, 1.0023, 0.9539, 0.9843, 
    1.1116, 0.9037, 1.0865, 1.0955, 
    1.0865, 1.1529, 1.0507
]

# n_lbls = 20
# s = [0] * n_lbls
# for i in range(len(gen_val)):
#     lbls = gen_val[i][1]
#     for j in range(n_lbls):
#         s[j] += lbls[..., j].sum()
# print(s)        
# for i in range(len(gen_train)):
#     lbls = gen_train[i][1]
#     for j in range(n_lbls):
#         s[j] += lbls[..., j].sum()
# print(s)
class_frequencies = [
    2381680967.0, 385659445.0, 1461641548.0, 42917813.0, 
    55975907.0, 81355164.0, 13285481.0, 36546566.0, 
    1038651996.0, 71574562.0, 252744993.0, 79239848.0, 
    9438758.0, 446059503.0, 17532539.0, 16553410.0, 
    13895603.0, 6178567.0, 29365708.0, 847304822.0
]
class_frequencies = [cf / 3475 / 1024 / 2048 for cf in class_frequencies]  # per pixel

# 2 * IoU * (1 + IoU) / (3 + IoU) for expected IoUs from some top submission
# roughly, weighs higher-expected-IoU classes more, since their incremental gains from getting a previously wrong pixel right are larger.
my_class_weights = [
    0.986388616484499,
    0.861620153626506,
    0.931111145930998,
    0.594947055526582,
    0.624917286319395,
    0.68796346303725,
    0.76164471681839,
    0.802451839949493,
    0.930431726613429,
    0.693919913768075,
    0.953283659512019,
    0.862984284311457,
    0.713803032412203,
    0.959006622659559,
    0.773685069664039,
    0.914790490449003,
    0.896106673708737,
    0.697513951168556,
    0.759241686259323,
]


def focal_loss(y_true, y_pred, gamma=2):
    # numerics
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    loss = 0
    for i in range(19):
        y_true_cls, y_pred_cls = y_true[..., i], y_pred[..., i]
        loss -= class_weights[i] * K.mean(y_true_cls * K.pow(1 - y_pred_cls, gamma) * K.log(y_pred_cls))# / class_frequencies[i]
    loss /= 19
    return loss

def my_loss(y_true, y_pred, gamma=0):
    ''' Balances FP and FN error with unbalanced classes, which no CCE formulation I found does (didn't search too hard because it's easy). '''
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = 0
    for i in range(19):
        y_true_cls, y_pred_cls = y_true[..., i], y_pred[..., i]
        # FN
        loss -= K.mean(y_true_cls * K.pow(1 - y_pred_cls, gamma) * K.log(y_pred_cls)) / class_frequencies[i]
        # FP
        loss -= K.mean((1 - y_true_cls) * K.pow(y_pred_cls, gamma) * K.log(1 - y_pred_cls)) / class_frequencies[i]
    return loss

def my_loss2(y_true, y_pred, gamma=0):
    ''' Balances FP and FN error with unbalanced classes, which no CCE formulation I found does (didn't search too hard because it's easy). '''
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = 0
    for i in range(19):
        y_true_cls, y_pred_cls = y_true[..., i], y_pred[..., i]
        # FN
        loss -= K.mean(y_true_cls * K.pow(1 - y_pred_cls, gamma) * K.log(y_pred_cls)) / class_frequencies[i] * my_class_weights[i]
        # FP
        loss -= K.mean((1 - y_true_cls) * K.pow(y_pred_cls, gamma) * K.log(1 - y_pred_cls)) / class_frequencies[i] * my_class_weights[i]
    return loss
