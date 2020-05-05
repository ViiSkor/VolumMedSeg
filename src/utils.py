import glob
import re
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm


def get_fpaths(data_dir):
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
  t1 = glob.glob(os.path.join(data_dir, '*/*t1.nii.gz'))
  t2 = glob.glob(os.path.join(data_dir, '*/*t2.nii.gz'))
  flair = glob.glob(os.path.join(data_dir, '*/*flair.nii.gz'))
  t1ce = glob.glob(os.path.join(data_dir, '*/*t1ce.nii.gz'))
  seg = glob.glob(os.path.join(data_dir, '*/*seg.nii.gz'))  # Ground Truth
  
  pat = re.compile('.*_(\w*)\.nii\.gz')

  data_paths = [{
      pat.findall(item)[0]:item
      for item in items
  }
  for items in list(zip(t1, t2, t1ce, flair, seg))]

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
