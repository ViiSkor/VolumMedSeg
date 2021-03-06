import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


plt.style.use('seaborn-pastel')


def animate_scan(images:list, titles:list, figsize=(16, 8)):
  fig = plt.figure(figsize=figsize)
  plots_num = len(titles)
  axis = [fig.add_subplot(1, plots_num, i) for i in range(1, plots_num+1)]

  myimages = []
  for i in range(images[0].shape[0]):
    curr_slice = []
    for ax_idx, (img, title) in enumerate(zip(images, titles)):
      axis[ax_idx].axis('off')
      axis[ax_idx].set_title(title, fontsize=18)
      curr_slice.append(axis[ax_idx].imshow(img[i], cmap='Greys_r'))
      
    myimages.append(curr_slice)

  anim = animation.ArtistAnimation(fig, myimages, interval=1000, blit=True, repeat_delay=1000)
  return anim


def show_class_frequency(classes_freq, classes2show, pixel2class):
  fig, ax = plt.subplots(figsize=(16,8))    
  height = [v for k, v in classes_freq.items() if pixel2class[k] in classes2show]
  bars = [pixel2class[k] for k in classes_freq.keys() if pixel2class[k] in classes2show]
  plt.bar(bars, height, color=[(0.1, 0.1, 0.1, 0.1) for _ in range(len(bars))], edgecolor='blue')
  plt.title('Class frequency', fontsize=25)
  plt.xlabel('classes', fontsize=20)
  plt.ylabel('counts', fontsize=20)
  for i, v in enumerate(height):
    ax.text(i-len(height)*0.05, v, str(v), fontweight='bold', fontsize=15)
  plt.show()


def show_images(images, cols = 1, scale=4, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.

    References:
        Adapted from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    """

    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    fig.tight_layout()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(np.ceil(n_images/float(cols)), cols, n + 1)
        a.axis('off')
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        #a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images / scale)
    plt.show()


def show_depth_hist(data):
  depths = []
  for modalities in data:
    depths.append(modalities['t1'].shape[0])

  plt.figure(figsize=(16,8))
  plt.title("Modalities' depth distribution", fontdict={'fontsize': 20})
  plt.hist(depths, bins=len(depths)//9)
  plt.show()


def show_modalities(imgs, slice_num, scan_types):
  fig, ax = plt.subplots(nrows=1, ncols=len(scan_types), figsize=(18, 9))

  for ax, scan in zip(ax.flat, scan_types):
      img = imgs[scan]
      ax.imshow(img[slice_num, :, :, 0], cmap='gray')
      ax.set_title(scan)
      ax.axis('off')

  plt.tight_layout(True)
  plt.show()
