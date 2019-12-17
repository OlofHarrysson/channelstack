import torch, argparse, math
from torch.optim.lr_scheduler import CosineAnnealingLR
import anyfig

from src.data.data import get_trainloader, get_valloader
from src.models.model import get_model
from src.config.config_classes import Cookie, Colab
from src.utils.utils import seed_program
from src.logger import Logger
from src.utils.utils import ProgressbarWrapper as Progressbar
from src.validator import Validator


def train(config):
  train_loader, val_loader = setup_dataloaders(config)
  model, optimizer, lr_scheduler, logger, validator = setup_train(config)

  # Init progressbar
  n_batches = len(train_loader)
  n_epochs = math.ceil(config.optim_steps / n_batches)
  pbar = Progressbar(n_epochs, n_batches)

  # Init variables
  optim_steps = 0
  val_freq = config.validation_freq

  # Training loop
  for epoch in pbar(range(1, n_epochs + 1)):
    for batch_i, data in enumerate(train_loader, 1):
      pbar.update(epoch, batch_i)

      # Validation
      if optim_steps % val_freq == 0:
        validator.validate(model, val_loader, optim_steps)

      inputs, labels = data
      outputs = model(inputs)
      loss, accuracy = model.calc_loss(outputs, labels, accuracy=True)
      loss.backward()
      optimizer.step()
      optim_steps += 1
      optimizer.zero_grad()

      # Decrease learning rate
      lr_scheduler.step()


def setup_dataloaders(config):
  train_loader = get_trainloader(config)
  val_loader = get_valloader(config)
  return train_loader, val_loader


def setup_train(config):
  model = get_model(config)
  optimizer = torch.optim.Adam(model.parameters(), lr=config.start_lr)
  lr_scheduler = CosineAnnealingLR(optimizer,
                                   T_max=config.optim_steps,
                                   eta_min=config.end_lr)
  logger = Logger(config)
  validator = Validator(config, logger)

  return model, optimizer, lr_scheduler, logger, validator


if __name__ == '__main__':
  config = anyfig.setup_config(default_config='Cookie')
  print(config)
  print('\n{}\n'.format(config.save_comment))
  seed_program(config.seed)
  train(config)
