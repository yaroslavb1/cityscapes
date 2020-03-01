import numpy as np
import tensorflow.keras as keras
import cv2
import glob
from tensorflow.keras.utils import to_categorical
import cityscapesScripts.cityscapesscripts.helpers.labels as labels

mapping_id_to_trainId = {label.id : label.trainId if label.trainId != 255 else 19 for label in labels.labels}

CITYSCAPES_MEAN = np.array([0.28689554, 0.32513303, 0.28389177])
CITYSCAPES_STD = np.array([0.18696375, 0.19017339, 0.18720214])


class CityScapesGenerator(keras.utils.Sequence):
    def __init__(self, subset, batch_size=1, crop=(512, 1024), resize=1, augment=True, dir_='.', shuffle=True):
        self.batch_size = batch_size
        self.crop = crop
        self.resize = resize
        self.augment = augment
        self.shuffle = shuffle

        assert subset in ('train', 'test', 'val')
        self.lbls = sorted(glob.glob(f'{dir_}/gtFine_trainvaltest/gtFine/{subset}/*/*_labelIds.png'))
        self.imgs = sorted(glob.glob(f'{dir_}/leftImg8bit_trainvaltest/leftImg8bit/{subset}/*/*_leftImg8bit.png'))
        
        self.on_epoch_end()
        
    def __len__(self):
        return (len(self.imgs) - 1) // self.batch_size + 1
    
    def __getitem__(self, idx):
        idxs = self.order[self.batch_size * idx : self.batch_size * (idx + 1)]
        
        batch_imgs = []
        batch_lbls = []
        for i in idxs:
            img = cv2.imread(self.imgs[i])
            img = img[..., ::-1]
            img = img.astype('float')
            lbl = cv2.imread(self.lbls[i], 0)
            lbl = lbl[..., None]
            img, lbl = self.process(img, lbl)
            batch_imgs.append(img)
            batch_lbls.append(lbl)
        batch_imgs = np.stack(batch_imgs, axis=0)
        batch_lbls = np.stack(batch_lbls, axis=0)
        return batch_imgs, batch_lbls
        
    def process(self, img, lbl):
        # normalization
        img = img / 255
        img = (img - CITYSCAPES_MEAN) / CITYSCAPES_STD
        # fliplr
        if self.augment and np.random.rand() > 0.5:
            img = img[:, ::-1, :]
            lbl = lbl[:, ::-1, :]
#         # intensity
#         if self.augment:
#             for ch in range(img.shape[-1]):
#                 s = 1.1 ** (np.random.rand() * 2 - 1)
#                 img[ch] *= s
        # resize
        r = self.resize
        if self.augment:
            r *= 2 ** (np.random.rand() * 2 - 1)
        if r != 1:
            img = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA)
            lbl = cv2.resize(lbl, None, fx=r, fy=r, interpolation=cv2.INTER_NEAREST)
        # crop
        if self.augment:
            hc, wc = self.crop
            hi, wi = lbl.shape[:2]
            if hi >= hc and wi >= wc:
                # y = np.random.randint(0, hi - hc + 1)
                # x = np.random.randint(0, wi - wc + 1)
                # changed to only pick one of the corners, to not undersample edges too much.
                y = np.random.choice([0, hi - hc])
                x = np.random.choice([0, wi - wc])
                img = img[y:y+hc, x:x+wc]
                lbl = lbl[y:y+hc, x:x+wc]
            else:
                raise Exception
        # to trainID
        lbl = np.vectorize(mapping_id_to_trainId.__getitem__)(lbl)
        # to categorical
        lbl = keras.utils.to_categorical(lbl, num_classes=20)
        return img, lbl
        
    def on_epoch_end(self):
        if self.shuffle:
            self.order = np.random.permutation(len(self.imgs))
        else:
            self.order = np.arange(len(self.imgs))
