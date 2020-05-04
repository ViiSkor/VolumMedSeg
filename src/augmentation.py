import numpy as np
import nibabel as nib
from scipy import ndimage


def crop(data, masks, depth=None, slice_shape=None):
  """Crop samples for a neural network input.

  Args:
    data (`numpy.array`): 
      numpy arrays [x, y, z, channels]/[x, y, channels]. Scan data.
    masks (`numpy.array`): 
      numpy arrays [x, y, z, channels]/[x, y, channels]. Ground truth data.
    depth (int): New z of a sample.
    slice_shape (tuple): New xy shape of a sample.
  Returns:
    data (`numpy.array`): Croped numpy arrays [x, y, z, channels]
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
        raise RuntimeError("unexpected dimension")

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
      numpy arrays [x, y, z, channels]/[x, y, channels]. Scan data.
    masks (`numpy.array`): 
      numpy arrays [x, y, z, channels]/[x, y, channels]. Ground truth data.
    prev_shape (tuple): Old shape of a sample
    shape (tuple): New shape of a sample.
    n_channels (int): The number of a case's channels/modalities/classes.
  Returns:
    new_data (`numpy.array`): Padded numpy data [x, y, z, channels]
    new_masks (`numpy.array`): Padded numpy ground truth [x, y, z, channels]
  """

  new_data = np.zeros((n_channels, *shape))
  new_masks = np.zeros((n_classes, *shape))
  start = (np.array(shape) / 2. - np.array(prev_shape) / 2.).astype(int)
  end = start + np.array([int(dim) for dim in prev_shape], dtype=int)
  if len(shape) == 2:
    new_data[start[0]:end[0], start[1]:end[1]] = data[:, :]
    new_masks[start[0]:end[0], start[1]:end[1]] = masks[:, :]
  elif len(shape) == 3:
    new_data[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]] = data[:, :, :, :]
    new_masks[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]] = masks[:, :, :, :]
  else:
    raise RuntimeError("unexpected dimension")
  return new_data, new_masks


def augment(data, masks, params):
  """Augment samples.

  Args:
    data (:obj:`numpy.array` of :obj:`np.float32`): 
      (x pathways) of numpy arrays [x, y, z, channels]. Scan data.
    masks (:obj:`numpy.array` of :obj:`np.int8`): 
      numpy arrays [x, y, z, channels]. Ground truth data.
    params (dict): None or Dictionary, with parameters of each augmentation type.
  Returns:
    data (:obj:`numpy.array` of :obj:`np.float32`): (x pathways) of np arrays [x, y, z, channels]
    masks (:obj:`numpy.array` of :obj:`np.int8`): np array of shape [x,y,z, classes]
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
        (x pathways) of np arrays [x, y, z, channels]. Scan data.
      masks (:obj:`numpy.array` of :obj:`np.int8`):
        numpy arrays [x, y, z, channels]. Ground truth data.
      n_dimensions (int): the number of dimensions
    Returns:
      data (:obj:`numpy.array` of :obj:`np.float32`): (x pathways) of np arrays [x, y, z, channels]
      masks (:obj:`numpy.array` of :obj:`np.int8`): np array of shape [x,y,z, classes]
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
        (x pathways) of np arrays [x, y, z, channels]. Scan data.
      shift (:obj:`dict` of :obj:`dict`): {'mu': 0.0, 'std':0.}
      params (:obj:`dict` of :obj:`dict`): {'mu': 1.0, 'std': '0.'}
    Returns:
      data (:obj:`numpy.array` of :obj:`np.float32`):
        (x pathways) of numpy arrays [x, y, z, channels]
        
    References:
        Adapted from https://github.com/deepmedic/deepmedic/blob/f937eaa79debf001db2df697ddb14d94e7757b9f/deepmedic/dataManagement/augmentSample.py#L23
    """
    
    n_channs = data[0].shape[-1]
    if shift is None:
        shift_per_chan = 0.
    elif shift['std'] != 0: # np.random.normal does not work for an std==0.
        shift_per_chan = np.random.normal(shift['mu'], shift['std'], [1, 1, 1, n_channs])
    else:
        shift_per_chan = np.ones([1, 1, 1, n_channs], dtype="float32") * shift['mu']
    
    if scale is None:
        scale_per_chan = 1.
    elif scale['std'] != 0:
        scale_per_chan = np.random.normal(scale['mu'], scale['std'], [1, 1, 1, n_channs])
    else:
        scale_per_chan = np.ones([1, 1, 1, n_channs], dtype="float32") * scale['mu']
    
    # Intensity augmentation
    for path_idx in range(len(data)):
        data[path_idx] = (data[path_idx] + shift_per_chan) * scale_per_chan
        
    return data


def random_rotate(data, masks, degrees=[-15, -10, -5, 0, 5, 10, 15]):
  """Rotate xy (width-height) by +-15/+-10/+-5 degrees.

  Args:
    data (:obj:`numpy.array` of :obj:`np.float32`): 
      (x pathways) of np arrays [x, y, z, channels]. Scan data.
    masks (:obj:`numpy.array` of :obj:`np.int8`):
      numpy arrays [x, y, z, channels]. Ground truth data.
    degrees (:obj:`numpy.array` of :obj:`int`): list of possible angle of rotation in degrees.
  Returns:
    data (:obj:`numpy.array` of :obj:`np.float32`): (x pathways) of np arrays [x, y, z, channels]
    masks (:obj:`numpy.array` of :obj:`np.int8`): np array of shape [x,y,z, classes]
  """

  degrees = np.random.choice(a=degrees, size=1)
  rot_deg = degrees[0]
  if rot_deg != 0:
    data = ndimage.rotate(data, rot_deg, reshape=False, axes=(1,2))
    masks = ndimage.rotate(masks, rot_deg, reshape=False, axes=(1,2))
    return data, masks
  return data, masks
