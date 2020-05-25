import numpy as np
import nibabel as nib
from scipy import ndimage


def augment(data, masks, params):
  """Augment samples.

  Args:
    data (:obj:`numpy.array` of :obj:`np.float32`): 
      (x pathways) of numpy arrays [channels, x, y, z]. Scan data.
    masks (:obj:`numpy.array` of :obj:`np.int8`): 
      numpy arrays [channels, x, y, z]. Ground truth data.
    params (dict): None or Dictionary, with parameters of each augmentation type.
  Returns:
    data (:obj:`numpy.array` of :obj:`np.float32`): (x pathways) of np arrays [channels, x, y, z]
    masks (:obj:`numpy.array` of :obj:`np.int8`): np array of shape [classes, x, y, z]
  """

  if params['hist_dist']:
    data = random_histogram_distortion(data, **params['hist_dist'])
  if params['flip']:
    data, masks = random_flip(data, masks, len(data))
  if params['rand_rot']:
    data, masks =  random_rotate(data, np.array(masks, dtype=np.uint8))
        
  return data, masks


def random_flip(data, masks, n_dimensions):
    """Flip (reflect) along axis.

    Args:
      data (:obj:`numpy.array` of :obj:`np.float32`): 
        (x pathways) of np arrays [channels, x, y, z]. Scan data.
      masks (:obj:`numpy.array` of :obj:`np.int8`):
        numpy arrays [channels, x, y, z]. Ground truth data.
      n_dimensions (int): the number of dimensions
    Returns:
      data (:obj:`numpy.array` of :obj:`np.float32`): (x pathways) of np arrays [channels, x, y, z]
      masks (:obj:`numpy.array` of :obj:`np.int8`): np array of shape [classes, x, y, z]
    """

    axis = [dim for dim in range(1, n_dimensions) if np.random.choice([True, False])]
    for axis_index in axis:
        data = np.flip(data, axis=axis_index)
        masks = np.flip(masks, axis=axis_index)

    return data, masks


def random_histogram_distortion(data: np.array, shift={'mu': 0.0, 'std': 0}, scale={'mu': 1.0, 'std': 0}):
    """Shift and scale the histogram of each channel.

    Args:
      data (:obj:`numpy.array` of :obj:`np.float32`): 
        (x pathways) of np arrays [channels, x, y, z]. Scan data.
      shift (:obj:`dict` of :obj:`dict`): {'mu': 0.0, 'std':0.}
      params (:obj:`dict` of :obj:`dict`): {'mu': 1.0, 'std': '0.'}
    Returns:
      data (:obj:`numpy.array` of :obj:`np.float32`):
        (x pathways) of numpy arrays [channels, x, y, z]
        
    References:
        Adapted from https://github.com/deepmedic/deepmedic/blob/f937eaa79debf001db2df697ddb14d94e7757b9f/deepmedic/dataManagement/augmentSample.py#L23
    """
    
    n_channs = data[0].shape[0]
    if len(data[0].shape) == 2:
      axis2distort = [1, 1]
    elif len(data[0].shape) == 3:
      axis2distort = [n_channs, 1, 1]
    elif len(data[0].shape) == 4:
      axis2distort = [n_channs, 1, 1, 1]
    else:
        raise RuntimeError(f"Got unexpected dimension {len(data[0].shape)}")

    if shift is None:
        shift_per_chan = 0.
    elif shift['std'] != 0: # np.random.normal does not work for an std==0.
        shift_per_chan = np.random.normal(shift['mu'], shift['std'], axis2distort)
    else:
        shift_per_chan = np.ones(axis2distort, dtype="float32") * shift['mu']
    
    if scale is None:
        scale_per_chan = 1.
    elif scale['std'] != 0:
        scale_per_chan = np.random.normal(scale['mu'], scale['std'], axis2distort)
    else:
        scale_per_chan = np.ones(axis2distort, dtype="float32") * scale['mu']
    
    # Intensity augmentation
    for path_idx in range(len(data)):
        data[path_idx] = (data[path_idx] + shift_per_chan) * scale_per_chan
        
    return data


def random_rotate(data, masks, degree_range=(-15, 15)):
  """Rotate xy (width-height) by +-15/+-10/+-5 degrees.

  Args:
    data (:obj:`numpy.array` of :obj:`np.float32`): 
      (x pathways) of np arrays [channels, x, y, z]. Scan data.
    masks (:obj:`numpy.array` of :obj:`np.int8`):
      numpy arrays [channels, x, y, z]. Ground truth data.
    degree_range (:obj:`tuple` of :obj:`int`): tuple of the possible range of rotation.
  Returns:
    data (:obj:`numpy.array` of :obj:`np.float32`): (x pathways) of np arrays [channels, x, y, z]
    masks (:obj:`numpy.array` of :obj:`np.int8`): np array of shape [classes, x, y, z]
  """

  degrees = np.arange(*degree_range, 1)
  degrees = np.random.choice(a=degrees, size=1)
  rot_deg = degrees[0]
  data = ndimage.rotate(data, rot_deg, reshape=False, axes=(1,2))
  masks = ndimage.rotate(masks, rot_deg, reshape=False, axes=(1,2))
  return data, masks
