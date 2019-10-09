from torchvision import datasets
from torch.utils.data import DataLoader
from ..transforms import Transformer


def get_trainloader(config):
  transformer = Transformer()
  transforms = transformer.get_train_transforms()
  dataset = MyCifar10('src/data/datasets', transforms, train=True)
  return DataLoader(dataset,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers)


def get_valloader(config):
  transformer = Transformer()
  transforms = transformer.get_val_transforms()
  dataset = MyCifar10('src/data/datasets', transforms, train=False)
  return DataLoader(dataset,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers)


class MyCifar10(datasets.CIFAR10):
  def __init__(self, path, transforms, train=True):
    super().__init__(path, train, download=True)
    self.transforms = transforms

  def __getitem__(self, index):
    im, label = super().__getitem__(index)
    return self.transforms(im), label