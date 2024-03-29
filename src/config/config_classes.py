import pyjokes
import random
from datetime import datetime as dtime
from collections import OrderedDict
import pprint
import anyfig


class DefaultConfig():
  def __init__(self):
    # ~~~~~~~~~~~~~~ General Parameters ~~~~~~~~~~~~~~

    # An optional comment to differentiate this run from others
    self.save_comment = pyjokes.get_joke()

    # Seed to create reproducable training results
    self.seed = random.randint(0, 2**32 - 1)

    # Start time to keep track of when the experiment was run
    self.start_time = dtime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Use GPU. Set to False to only use CPU
    self.use_gpu = True

    # Threads to use in data loading
    self.num_workers = 0

    # Batch size going into the network
    self.batch_size = 64

    # Using a pretrained network
    self.pretrained = False

    # Start and end learning rate for the scheduler
    self.start_lr = 1e-2
    self.end_lr = 1e-3

    # For how many steps to train
    self.optim_steps = 10000

    # How often to validate
    self.validation_freq = 100

    # How often to validate
    self.max_val_batches = 9999999

    self.im_channels = 3


@anyfig.config_class
class Cookie(DefaultConfig):
  def __init__(self):
    super().__init__()
    self.use_gpu = False
    self.validation_freq = 5
    self.max_val_batches = 10
    self.batch_size = 2


@anyfig.config_class
class Colab(DefaultConfig):
  def __init__(self):
    super().__init__()
    self.validation_freq = 500
    self.num_workers = 12
