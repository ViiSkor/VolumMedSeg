import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def animate_scan(scan, mask):
  fig = plt.figure(figsize=(16, 8))
  ax1 = fig.add_subplot(1,2,1)
  ax2 = fig.add_subplot(1,2,2)

  myimages = []
  for i in range(scan.shape[0]):
      ax1.axis('off')
      ax1.set_title('Scan', fontsize='medium')
      ax2.axis('off')
      ax2.set_title('Mask', fontsize='medium')
      
      myimages.append([ax1.imshow(scan[i], cmap='Greys_r'), ax2.imshow(mask[i], cmap='Greys_r')])

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


# Source https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
def show_images(images, cols = 1, scale=4, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
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