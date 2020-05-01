import nibabel as nib
import numpy as np


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
        Ground truth numpy arrays [x, y, z, classes]. Whole volumes, channels of a case.
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
        ncr = resize(ncr, out_shape, mode=mode)
        ed = resize(ed, out_shape, mode=mode)
        et = resize(et, out_shape, mode=mode)

    output = []
    if 'ncr' in output_classes:
      output.append(ncr)
    if 'ed' in output_classes:
      output.append(ed)
    if 'et' in output_classes:
      output.append(et)

    return np.array(output, dtype=np.uint8)


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


def preprocesse(imgs, dataset_name, dist_dir_path):
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

    img_flair = img_flair.get_data()
    img_t1 = img_t1.get_data()
    img_t1ce = img_t1ce.get_data()
    img_t2 = img_t2.get_data()
    gth = gth.get_data()


    # Crop the volumes
    # First, identify the max dimensions and then crop
    wi_st, wi_en, hi_st, hi_en, ch_st, ch_en = cropVolumes(img_flair, img_t1, img_t1ce, img_t2)
    img_flair = img_flair[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    img_t1 = img_t1[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    img_t1ce = img_t1ce[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    img_t2 = img_t2[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    gth = gth[wi_st:wi_en, hi_st:hi_en, ch_st:ch_en]
    

    # save the cropped volumes
    flair_cropped = nib.Nifti1Image(img_flair, affine_flair, header_flair)
    t1_cropped = nib.Nifti1Image(img_t1, affine_t1, header_t1)
    t1ce_cropped = nib.Nifti1Image(img_t1ce, affine_t1ce, header_t1ce)
    t2_cropped = nib.Nifti1Image(img_t2, affine_t2, header_t2)
    gth_cropped = nib.Nifti1Image(gth, affine_gth, header_gth)

    
    # create the directories if they do not exist
    dist_dir_path = dist_dir_path + os.sep + imgs['flair'].split('/')[1]
    if not os.path.isdir(dist_dir_path):
        os.makedirs(dist_dir_path)

    nib.save(flair_cropped, dist_dir_path + os.sep + imgs['flair'].split('/')[-1])
    nib.save(t1_cropped, dist_dir_path + os.sep + imgs['t1'].split('/')[-1])
    nib.save(t1ce_cropped, dist_dir_path + os.sep + imgs['t1ce'].split('/')[-1])
    nib.save(t2_cropped, dist_dir_path + os.sep + imgs['t2'].split('/')[-1])
    nib.save(gth_cropped, dist_dir_path + os.sep + imgs['seg'].split('/')[-1])
  