from torchvision import transforms
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image


class Transformer():
  def __init__(self):
    self.return_im = []
    self.add_return_im = lambda im: self.return_im.extend(im)

    # Augmentation mode
    self.aug_fn = self.rgb
    # self.aug_fn = self.rgbnoise
    # self.aug_fn = self.rgbmore

  def get_train_transforms(self):
    return transforms.Compose([
      self.aug_fn,
      transforms.ToTensor(),
    ])

  def get_val_transforms(self):
    return self.get_train_transforms()

  def channel_info(self):
    pass

  def __call__(self, im):
    self.return_im = []
    return self.aug_fn(im)

  def rgb(self, im):
    return np.array(im)

  def rgbnoise(self, im):
    im_size = im.size
    noise_layer = np.random.randint(0, 255, im_size, dtype='uint8')
    return np.dstack((np.array(im), noise_layer))

  def rgbmore(self, im):
    return_im = []
    add_return_im = lambda im: return_im.extend(im)

    grey = np.array(im.convert(mode='L'))
    im = np.array(im)
    rgb_grey = np.dstack((im, grey))

    edge = iaa.EdgeDetect(alpha=1)(images=rgb_grey)

    dir_edge = lambda d: iaa.DirectedEdgeDetect(alpha=1, direction=d)(images=
                                                                      grey)
    dir_edges = np.array(
      [dir_edge(d) for d in np.linspace(0, 1, num=3, endpoint=False)])
    dir_edges = np.transpose(dir_edges, (1, 2, 0))
    canny = iaa.Canny(alpha=1.0,
                      hysteresis_thresholds=128,
                      sobel_kernel_size=4,
                      deterministic=True,
                      colorizer=iaa.RandomColorsBinaryImageColorizer(
                        color_true=255, color_false=0))(images=grey)

    avg_pool = iaa.AveragePooling(2)(images=grey)
    max_pool = iaa.MaxPooling(2)(images=grey)
    min_pool = iaa.MinPooling(2)(images=grey)

    add_return_im([im, grey])
    add_return_im([edge, dir_edges, canny])
    add_return_im([avg_pool, max_pool, min_pool])
    return np.dstack(return_im)


if __name__ == '__main__':
  from logger import Logger
  logger = Logger(config=None)

  transformer = Transformer()
  im = ia.quokka(size=(256, 256))
  im = Image.fromarray(im)
  aug_im = transformer(im)

  aug_im = np.transpose(aug_im, (2, 0, 1))
  aug_im = np.expand_dims(aug_im, axis=1)
  logger.log_channels(aug_im)
