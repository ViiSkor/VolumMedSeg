import numpy as np
import nibabel as nib
from tensorflow.keras.utils import Sequence

from augmentation import augment
from preprocessing import pad, crop, preprocess_label


class Med3DDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_fpaths,
                 batch_size=8, dim=(144, 192, 160),
                 scan_types=['t1'],
                 output_classes=['ncr'],
                 merge_classes=False,
                 shuffle=False,
                 hist_dist=False,
                 flip=False,
                 rand_rot=False):
        'Initialization'
        self.batch_size = batch_size
        self.dim = dim
        self.dim_before_axes_swap = (dim[-1], dim[1], dim[0])
        self.list_fpaths = list_fpaths
        self.merge_classes = merge_classes
        self.scan_types = scan_types
        self.output_classes = output_classes
        self.n_channels = len(scan_types)
        self.n_classes = len(output_classes)
        if self.merge_classes:
          self.n_classes = 1
        self.shuffle = shuffle
        self.augment_params = {'hist_dist': None, 
                               'flip': flip,
                               'rand_rot': rand_rot}
        if hist_dist:
          self.augment_params['hist_dist'] = {
                              'shift': {
                                  'mu': 0, 
                                  'std': 0.25
                                  },
                              'scale': {
                                  'mu': 1, 
                                  'std': 0.25
                                  }
                                }
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_fpaths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_fpaths_temp = [self.list_fpaths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_fpaths_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_fpaths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_fpaths_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=np.float32)

        # Generate data
        for i, imgs in enumerate(list_fpaths_temp):
          modalities = np.array([np.asanyarray(nib.load(imgs[m]).dataobj) for m in self.scan_types])
          masks = preprocess_label(np.asanyarray(nib.load(imgs['seg']).dataobj),
                                   output_classes=self.output_classes,
                                   merge_classes=self.merge_classes)

          modalities, masks = crop(modalities, masks, depth=self.dim[0])
          modalities, masks = pad(modalities, masks, masks.shape[1:], 
                                  self.dim_before_axes_swap,
                                  self.n_channels,
                                  self.n_classes)
          
          modalities, masks = augment(modalities, masks, self.augment_params)
          
          modalities = np.moveaxis(modalities, 0, -1)
          masks = np.moveaxis(masks, 0, -1)

          X[i] = np.swapaxes(modalities, 0, -2)
          y[i] = np.swapaxes(masks, 0, -2)

          X[i] = (X[i] - np.mean(X[i])) / np.std(X[i])

        return X, y


class Med2DDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_fpaths,
                 batch_size=8, dim=(192, 160),
                 scan_types=['t1'],
                 output_classes=['ncr'],
                 merge_classes=False,
                 shuffle=False,
                 hist_dist=False,
                 flip=False,
                 rand_rot=False):
        'Initialization'
        self.batch_size = batch_size
        self.dim = dim
        self.list_fpaths = list_fpaths
        self.merge_classes = merge_classes
        self.scan_types = scan_types
        self.output_classes = output_classes
        self.n_channels = len(scan_types)
        self.n_classes = len(output_classes)
        if self.merge_classes:
          self.n_classes = 1
        self.shuffle = shuffle
        self.augment_params = {'hist_dist': None, 
                               'flip': flip,
                               'rand_rot': rand_rot}
        if hist_dist:
          self.augment_params['hist_dist'] = {
                              'shift': {
                                  'mu': 0, 
                                  'std': 0.25
                                  },
                              'scale': {
                                  'mu': 1, 
                                  'std': 0.25
                                  }
                                }
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_fpaths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_fpaths_temp = [self.list_fpaths[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_fpaths_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_fpaths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_fpaths_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=np.float32)

        # Generate data
        for i, imgs in enumerate(list_fpaths_temp):
          curr_slice = imgs['seg'][1]
          modalities = np.array([np.load(imgs[m][0])[curr_slice] for m in self.scan_types])
          masks = preprocess_label(np.load(imgs['seg'][0])[curr_slice],
                                   output_classes=self.output_classes,
                                   merge_classes=self.merge_classes)

          modalities, masks = pad(modalities, masks, masks.shape[1:], 
                                  self.dim,
                                  self.n_channels,
                                  self.n_classes)
          
          modalities, masks = augment(modalities, masks, self.augment_params)
          
          X[i] = np.moveaxis(modalities, 0, -1)
          y[i] = np.moveaxis(masks, 0, -1)

          X[i] = (X[i] - np.mean(X[i])) / np.std(X[i])

        return X, y
  
  