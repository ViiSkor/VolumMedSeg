import os
import nibabel as nib
import numpy as np
from tqdm import tqdm


def fill_labels(img, slice_nums):
    kernel = np.ones((3, 3))
    img = img.astype(np.float32)
    for i in range(slice_nums):
      img_slice = img[i, :, :]
      img_slice_closed = cv2.morphologyEx(img_slice, cv2.MORPH_CLOSE, kernel, iterations=3)
      img[i] = img_slice_closed
    return img.astype(np.uint8)


def preprocess_label(mask, output_classes=['ed'], merge_classes=False, out_shape=None, mode='nearest'):
    """Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)

    Args:
      mask (numpy.array): 
        Ground truth numpy arrays [classes, x, y, z]. Whole volumes, channels of a case.
      output_classes (:obj:`list` of :obj:`str`): classes to sepatare.
      merge_classes (bool): Merge output_classes into one or not.
      out_shape (tuple): Shape for scaling ground truth labels.
      mode (str): Resizeing mode.
    Returns:
      (numpy.array): Separated binarized ground truth labels.
    """

    ncr = mask == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = mask == 2  # Peritumoral Edema (ED)
    et = mask == 4  # GD-enhancing Tumor (ET)
    
    if out_shape is not None:
        ncr = fill_labels(resize(ncr, out_shape, mode=mode), slice_nums=out_shape[-1])
        ed = fill_labels(resize(ed, out_shape, mode=mode), slice_nums=out_shape[-1])
        et = fill_labels(resize(et, out_shape, mode=mode), slice_nums=out_shape[-1])

    masks = []
    if 'ncr' in output_classes:
      masks.append(ncr)
    if 'ed' in output_classes:
      masks.append(ed)
    if 'et' in output_classes:
      masks.append(et)
    
    output = []
    if merge_classes:
      output = np.zeros_like(masks[0])
      for label in masks:
        output += label
      output = [output]
    else:
      output = masks

    return np.array(output, dtype=np.uint8)


def crop(data, masks, depth=None, slice_shape=None):
  """Crop samples for a neural network input.

  Args:
    data (`numpy.array`): 
      numpy arrays [channels, x, y, z]/[channels, x, y]. Scan data.
    masks (`numpy.array`): 
      numpy arrays [channels, x, y, z]/[channels, x, y]. Ground truth data.
    depth (int): New z of a sample.
    slice_shape (tuple): New xy shape of a sample.
  Returns:
    data (`numpy.array`): Croped numpy arrays [channels, x, y, z]
  """


  if slice_shape:
      if len(data.shape) == 3:
        vertical_shift = int((data.shape[0] - slice_shape[0]) // 2)
        horizontal_shift = int((data.shape[1] - slice_shape[1]) // 2)
        data = data[vertical_shift:slice_shape[0]+vertical_shift,horizontal_shift:slice_shape[1]+horizontal_shift,:]
      elif len(data.shape) == 4:
        vertical_shift = int((data.shape[1] - slice_shape[0]) // 2)
        horizontal_shift = int((data.shape[2] - slice_shape[1]) // 2)
        data = data[vertical_shift:slice_shape[0]+vertical_shift,horizontal_shift:slice_shape[1]+horizontal_shift,:]
      else:
        raise RuntimeError(f"Got unexpected dimension {len(data.shape)}")

  if depth:
    if depth < data.shape[-1]:
      if len(data.shape) == 4:
        depth_shift = int((data.shape[-1] - depth) // 2)
        data = data[:, :,:,depth_shift:depth+depth_shift]
        masks = masks[:, :,:,depth_shift:depth+depth_shift]

  return data, masks


def pad(data, masks, prev_shape, shape, n_channels, n_classes):
  """Pad samples for a neural network input.

  Args:
    data (`numpy.array`): 
      numpy arrays [channels, x, y, z]/[channels, x, y]. Scan data.
    masks (`numpy.array`): 
      numpy arrays [channels, x, y, z]/[channels, x, y]. Ground truth data.
    prev_shape (tuple): Old shape of a sample
    shape (tuple): New shape of a sample.
    n_channels (int): The number of a case's channels/modalities/classes.
  Returns:
    new_data (`numpy.array`): Padded numpy data [channels, x, y, z]
    new_masks (`numpy.array`): Padded numpy ground truth [channels, x, y, z]
  """

  new_data = np.zeros((n_channels, *shape))
  new_masks = np.zeros((n_classes, *shape))
  start = (np.array(shape) / 2. - np.array(prev_shape) / 2.).astype(int)
  end = start + np.array([int(dim) for dim in prev_shape], dtype=int)
  if len(shape) == 2:
    new_data[:, start[0]:end[0], start[1]:end[1]] = data[:, :, :]
    new_masks[:, start[0]:end[0], start[1]:end[1]] = masks[:, :, :]
  elif len(shape) == 3:
    new_data[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]] = data[:, :, :, :]
    new_masks[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]] = masks[:, :, :, :]
  else:
    raise RuntimeError(f"Got unexpected dimension {len(shape)}")
  return new_data, new_masks



def prepare(data_paths:dict, dataset_name:str, preprocessed_dist:str, mode="3D"):
  for i, imgs in enumerate(tqdm(data_paths)):
      preprocesse(imgs, dataset_name, preprocessed_dist, mode)


# Source: https://github.com/sacmehta/3D-ESPNet/blob/master/utils.py
def cropVolume(img, data=False):
    '''
    Helper function to remove the redundant black area from the 3D volume
    :param img: 3D volume
    :param data: Nib allows you to access 3D volume data using the get_data(). If you have already used it before
    calling this function, then it is false
    :return: returns the crop positions acrss 3 axes (channel, width and height)
    '''
    if not data:
       img = img.get_data()
    sum_array = []


    for ch in range(img.shape[2]):
        values, indexes = np.where(img[:, :, ch] > 0)
        sum_val = sum(values)
        sum_array.append(sum_val)
    ch_s = np.nonzero(sum_array)[0][0]
    ch_e = np.nonzero(sum_array)[0][-1]
    sum_array = []
    for width in range(img.shape[0]):
        values, indexes = np.where(img[width, :, :] > 0)
        sum_val = sum(values)
        sum_array.append(sum_val)
    wi_s = np.nonzero(sum_array)[0][0]
    wi_e = np.nonzero(sum_array)[0][-1]
    sum_array = []
    for width in range(img.shape[1]):
        values, indexes = np.where(img[:, width, :] > 0)
        sum_val = sum(values)
        sum_array.append(sum_val)
    hi_s = np.nonzero(sum_array)[0][0]
    hi_e = np.nonzero(sum_array)[0][-1]

    return ch_s, ch_e, wi_s, wi_e, hi_s, hi_e


def cropVolumes(img1, img2, img3, img4):
    '''
        This function crop the 4 volumes that BRATS dataset provides
        :param img1: Volume 1
        :param img2: Volume 2
        :param img3: Volume 3
        :param img4: Volume 4
        :return: maximum dimensions across three dimensions
    '''
    ch_s1, ch_e1, wi_s1, wi_e1, hi_s1, hi_e1 = cropVolume(img1, True)
    ch_s2, ch_e2, wi_s2, wi_e2, hi_s2, hi_e2 = cropVolume(img2, True)
    ch_s3, ch_e3, wi_s3, wi_e3, hi_s3, hi_e3 = cropVolume(img3, True)
    ch_s4, ch_e4, wi_s4, wi_e4, hi_s4, hi_e4 = cropVolume(img4, True)

    ch_st = min(ch_s1, ch_s2, ch_s3, ch_s4)
    ch_en = max(ch_e1, ch_e2, ch_e3, ch_e4)
    wi_st = min(wi_s1, wi_s2, wi_s3, wi_s4)
    wi_en = max(wi_e1, wi_e2, wi_e3, wi_e4)
    hi_st = min(hi_s1, hi_s2, hi_s3, hi_s4)
    hi_en = max(hi_e1, hi_e2, hi_e3, hi_e4)
    return wi_st, wi_en, hi_st, hi_en, ch_st, ch_en


def save_nifti(imgs2save):
  for imgs in imgs2save:
    nib.save(*imgs)


def save_npy(imgs2save):
  frst_slice = 0
  last_slice = 0
  seg = np.swapaxes(imgs2save["seg"]["modality"], 0, -1)
  for i in range(seg.shape[0]):
    curr_slice = seg[i, :, :]
    if np.sum(curr_slice) == 0:
      if last_slice <= frst_slice:
        frst_slice = i
    else:
      last_slice = i
  frst_slice += 1

  for name, data in imgs2save.items():
    modality = data["modality"]
    path = data["path"]
    modality = np.swapaxes(modality, 0, -1)
    modality = modality[frst_slice:last_slice]
    for i in range(modality.shape[0]):
      curr_slice = modality[i, :, :]
      if not os.path.isdir(path):
        os.makedirs(path)
      slice_dist_path = path + os.sep + str(i)
      with open(f"{slice_dist_path}.npy", "wb") as f:
            np.save(f, curr_slice)  


def preprocesse(imgs, dataset_name, dist_dir_path, mode="3D"):
    """Preprocesse nii.gz data.

    Args:
      imgs (:obj:`tuple` of :obj:`dict`):
        Tuple of the dictionary for each patient with structure:
          {
            't1': <path to t1 MRI file>
            't2': <path to t2 MRI>
            'flair': <path to FLAIR MRI file>
            't1ce': <path to t1ce MRI file>
            'seg': <path to Ground Truth file>
        }
      dataset_name (str): Dataset's name.
      dist_dir_path (str): The destination path of preprocessed data.
    """

    img_flair = nib.load(imgs['flair'])
    affine_flair = img_flair.affine
    header_flair = img_flair.header

    img_t1 = nib.load(imgs['t1'])
    affine_t1 = img_t1.affine
    header_t1 = img_t1.header

    img_t1ce = nib.load(imgs['t1ce'])
    affine_t1ce = img_t1ce.affine
    header_t1ce = img_t1ce.header

    img_t2 = nib.load(imgs['t2'])
    affine_t2 = img_t2.affine
    header_t2 = img_t2.header

    gth = nib.load(imgs['seg'])
    affine_gth = gth.affine
    header_gth = gth.header

    img_flair = np.asanyarray(img_flair.dataobj)
    img_t1 = np.asanyarray(img_t1.dataobj)
    img_t1ce = np.asanyarray(img_t1ce.dataobj)
    img_t2 = np.asanyarray(img_t2.dataobj)
    gth = np.asanyarray(gth.dataobj)


    # Crop the volumes
    # First, identify the max dimensions and then crop
    wi_st, wi_en, hi_st, hi_en, ch_st, ch_en = cropVolumes(img_flair, img_t1, img_t1ce, img_t2)
    img_flair = img_flair[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    img_t1 = img_t1[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    img_t1ce = img_t1ce[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    img_t2 = img_t2[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    gth = gth[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]

    
    # create the directories if they do not exist
    dist_dir_path = dist_dir_path + os.sep + imgs['flair'].split('/')[-2]
    if not os.path.isdir(dist_dir_path):
      os.makedirs(dist_dir_path)

    if mode=="3D":
      # save the cropped volumes
      flair_cropped = nib.Nifti1Image(img_flair, affine_flair, header_flair)
      t1_cropped = nib.Nifti1Image(img_t1, affine_t1, header_t1)
      t1ce_cropped = nib.Nifti1Image(img_t1ce, affine_t1ce, header_t1ce)
      t2_cropped = nib.Nifti1Image(img_t2, affine_t2, header_t2)
      gth_cropped = nib.Nifti1Image(gth, affine_gth, header_gth)

      imgs2save = [
        (flair_cropped, dist_dir_path + os.sep + imgs['flair'].split('/')[-1]),
        (t1_cropped, dist_dir_path + os.sep + imgs['t1'].split('/')[-1]),
        (t1ce_cropped, dist_dir_path + os.sep + imgs['t1ce'].split('/')[-1]),
        (t2_cropped, dist_dir_path + os.sep + imgs['t2'].split('/')[-1]),
        (gth_cropped, dist_dir_path + os.sep + imgs['seg'].split('/')[-1])
      ]
      save_nifti(imgs2save)
    elif mode=="2D":
      imgs2save = {
        "flair": {"modality": img_flair, "path": dist_dir_path + os.sep + imgs['flair'].split('/')[-1].split('.')[-3]},
        "t1": {"modality": img_t1, "path": dist_dir_path + os.sep + imgs['t1'].split('/')[-1].split('.')[-3]},
        "t1ce": {"modality": img_t1ce, "path": dist_dir_path + os.sep + imgs['t1ce'].split('/')[-1].split('.')[-3]},
        "t2": {"modality": img_t2, "path": dist_dir_path + os.sep + imgs['t2'].split('/')[-1].split('.')[-3]},
        "seg": {"modality": gth, "path": dist_dir_path + os.sep + imgs['seg'].split('/')[-1].split('.')[-3]}
      }
      save_npy(imgs2save)
    else:
      raise ValueError(f"mode must be one of ['2D', '3D'], got {mode}")
