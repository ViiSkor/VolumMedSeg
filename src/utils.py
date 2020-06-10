import glob
import re
import os
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm

from preprocessing import preprocess_label, crop, pad


def get_fpaths(data_dir, mode="3D"):
  '''Parse all the filenames and create a dictionary for each patient with structure:
  {
      't1': <path to t1 MRI file>
      't2': <path to t2 MRI>
      'flair': <path to FLAIR MRI file>
      't1ce': <path to t1ce MRI file>
      'seg': <path to Ground Truth file>
  }
  '''

  # Get a list of files for all modalities individually
  if mode == "3D":
    ext = 'nii.gz'
    pat = re.compile('.*_(\w*)\.nii\.gz')
  elif mode == "2D":
    ext = 'npy'
    pat = re.compile('.*_(\w*)\.npy')
  
  t1 = glob.glob(os.path.join(data_dir, f'*/*t1.{ext}'))
  t2 = glob.glob(os.path.join(data_dir, f'*/*t2.{ext}'))
  flair = glob.glob(os.path.join(data_dir, f'*/*flair.{ext}'))
  t1ce = glob.glob(os.path.join(data_dir, f'*/*t1ce.{ext}'))
  seg = glob.glob(os.path.join(data_dir, f'*/*seg.{ext}'))  # Ground Truth

  data_paths = [{
      pat.findall(item)[0]:item
      for item in items
  }
  for items in list(zip(t1, t2, t1ce, flair, seg))]

  return data_paths


def unpack_2D_fpaths(packed_data_paths, only_with_mask=True):
  upacked_data_paths = []
  mod_names = packed_data_paths[0].keys()
  for paths in packed_data_paths:
    frst_slice = 0
    last_slice = 0
    img = np.load(paths['seg'])
    if only_with_mask:
      for i in range(img.shape[0]):
        curr_slice = img[i, :, :]
        if np.sum(curr_slice) == 0:
          if last_slice <= frst_slice:
            frst_slice = i
        else:
          last_slice = i
      frst_slice += 1
    else:
      last_slice = img.shape[0]
    depth = 0
    modalities = {}
    for name, path in paths.items():
      data = np.load(path)
      data = data[frst_slice:last_slice]
      depth = data.shape[0]
      for i in range(depth):
        modalities[name] = modalities.get(name, []) + [(path, i)]

    for i in range(depth):
      upacked_data_paths.append({name: modalities[name][i] for name in mod_names})
      
  return upacked_data_paths


def get_data_paths4existing_slit(data_dir, splitted_data, mode="2D"):
  data_paths = []
  for modalities in splitted_data:
    curr_case = {}
    for name, path in modalities.items():
      path = path.split('/')[-2:]
      if mode=="2D":
        ext = '.npy'
      elif mode=="3D":
        ext = '.nii.gz'
      path[-1] = path[-1].split('.')[0] + ext
      curr_case[name] = os.path.join(data_dir, *path)
    data_paths.append(curr_case)
  return data_paths
def change_orientation(img):
  img = np.moveaxis(img, 0, -1)
  img = np.swapaxes(img, 0, -2)
  return img


def read_img(img_path):
  img = np.array([np.asanyarray(nib.load(img_path).dataobj)])
  img = change_orientation(img)
  return img


def get_nearest_multipleof_n(num, n):
  return num + (n - num % n)


def get_final_shape(data, multiple_of=16):
  types = ['t1', 't2', 't1ce', 'flair', 'seg']
  img_type_shapes = {t:[] for t in types}

  depth = 0
  height = 0
  width = 0
  for imgs in tqdm(data):
    for img_type in types:
      shape = imgs[img_type].shape
      depth = shape[0] if shape[0] > depth else depth
      height = shape[1] if shape[1] > height else height
      width = shape[2] if shape[2] > width else width

  if depth % 16 != 0:
    depth = get_nearest_multipleof_n(depth, multiple_of)
  if height % 16 != 0:
    height = get_nearest_multipleof_n(height, multiple_of)
  if width % 16 != 0:
    width = get_nearest_multipleof_n(width, multiple_of)

  return (depth, height , width)


def get_class_frequency(data_paths, classes):
  classes_freq = {c:0 for c in classes}
  for imgs in tqdm(data_paths):
    seg = read_img(imgs['seg']).flatten()
    (unique, counts) = np.unique(read_img(imgs['seg']), return_counts=True)
    for u, c in zip(unique, counts):
      classes_freq[u] += c
  return classes_freq


def calculate_class_weights(classes_freq, classes2calculate, pixel2class):
  freq_sum = sum([f for c, f in classes_freq.items() if pixel2class[c] in classes2calculate])
  # inverse probability weighting
  inverse_props = {pixel2class[c]: 1/(f/freq_sum) for c, f in classes_freq.items() if pixel2class[c] in classes2calculate}
  return {c: p/sum([other_p for other_c, other_p in inverse_props.items() if c != other_p]) for c, p in inverse_props.items()}


def get_preprocessed_data(data_paths:dict, scan_types=['t1', 'seg']):
  data = []
  for imgs in tqdm(data_paths):
    scans = {m:read_img(imgs[m]) for m in scan_types}
    data.append(scans)

  return data


def get_dataset_split(data_paths, train_ratio=0.7, seed=42, shuffle=True):
  random.seed(seed)

  n_samples = len(data_paths)
  n_train = int(n_samples*train_ratio)
  n_test = int(n_samples*(train_ratio-1)/2)

  if shuffle:
    random.shuffle(data_paths)

  train_data_paths = data_paths[:n_train]
  test_data_paths = data_paths[n_train:]
  val_data_paths = test_data_paths[:n_test]
  test_data_paths = test_data_paths[n_test:]

  return train_data_paths, test_data_paths, val_data_paths


def stack_2D_2_3D(samples_sep, arr, dim, n_channels):
  stacked_pred = []
  prev_idx = 0
  for sep in samples_sep:
    pred = arr[prev_idx:prev_idx+sep]
    if pred.shape[0] > dim[0]:
      depth_shift = int((pred.shape[0] - dim[0]) // 2)+1
      pred = pred[depth_shift:-depth_shift,:,:,:]
    pred = np.moveaxis(pred, -1, 0)
    _, pred = pad(pred, pred, pred.shape[1:],  dim, n_channels, n_channels)
    pred = np.moveaxis(pred, 0, -1)
    stacked_pred.append(pred)
    prev_idx += sep + 1

  return np.array(stacked_pred)


def load_sample(path, dim, scan_types, classes, merge_classes, n_channels, n_classes, mode="3D"):
  dim_before_axes_swap = (dim[-1], dim[1], dim[0])
  if mode=="3D":
    masks = preprocess_label(np.asanyarray(nib.load(path['seg']).dataobj), output_classes=classes, merge_classes=merge_classes)
    imgs = np.array([np.asanyarray(nib.load(path[m]).dataobj) for m in scan_types])
  elif mode=="2D":
    masks = preprocess_label(np.load(path['seg']), output_classes=classes, merge_classes=merge_classes)
    imgs = np.array([np.load(path[m]) for m in scan_types], dtype=np.float16)
    imgs = np.moveaxis(imgs, [0, 1, 2, 3], [0, 3, 2, 1])
    masks = np.moveaxis(masks, [0, 1, 2, 3], [0, 3, 2, 1])

  imgs, masks = crop(imgs, masks, depth=dim[0])
  imgs, masks = pad(imgs, masks, masks.shape[1:],  dim_before_axes_swap, n_channels, n_classes)

  imgs = change_orientation(imgs)
  masks = change_orientation(masks)

  return masks, imgs


def evaluate(data_paths, prediction, metric, dim, scan_types, classes, merge_classes, mode="3D"):
  scores = {'class': [], 'score': []}
  for path, pred in zip(data_paths, prediction):
    if merge_classes:
      load = ['mask']
    else:
      load = classes
    for cls in load:
      if merge_classes:
        cls = classes
        cls_name = 'mask'
      else:
        cls_name = cls
        cls = [cls]
      mask, _ = load_sample(path=path, dim=dim, scan_types=scan_types, classes=cls, merge_classes=merge_classes, n_channels=len(scan_types), n_classes=1, mode=mode)
      mask = np.array([mask])
      pred = np.array([pred])
      score = metric(mask, pred)
      scores['class'] = scores['class'] + [cls_name]
      scores['score'] = scores['score'] + [score.numpy()]
  return scores


def get_data_paths4existing_slit(data_dir, splitted_data, mode="2D"):
  data_paths = []
  for modalities in splitted_data:
    curr_case = {}
    for name, path in modalities.items():
      path = path.split('/')[-2:]
      if mode=="2D":
        ext = '.npy'
      elif mode=="3D":
        ext = '.nii.gz'
      path[-1] = path[-1].split('.')[0] + ext
      curr_case[name] = os.path.join(data_dir, *path)
    data_paths.append(curr_case)
  return data_paths


def get_2D_sep_slices(paths_unnpacked):
  samples_sep = []
  prev_slice_idx = 0
  for i, path in enumerate(paths_unnpacked):
    if path['seg'][1] == 0:
      samples_sep.append(prev_slice_idx)
    prev_slice_idx = path['seg'][1]

  samples_sep.append(prev_slice_idx)
  samples_sep.pop(0)
  return samples_sep