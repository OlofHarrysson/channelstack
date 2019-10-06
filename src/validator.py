import torch
import torch.nn as nn


class Validator():
  def __init__(self, config, logger):
    self.config = config
    self.logger = logger

  def validate(self, model, val_loader, step):
    corrects = torch.tensor([], dtype=bool)
    for batch_i, data in enumerate(val_loader, 1):
      if batch_i > self.config.max_val_batches:
        break

      inputs, labels = data
      preds = model.predict(inputs)
      corrects = torch.cat((corrects, preds.cpu() == labels))

    accuracy = corrects.sum(dtype=float) / len(corrects)
    self.logger.log_accuracy(accuracy, step)
    channel_amplitudes = first_layer_weights(model)
    self.logger.log_first_layer(channel_amplitudes, step)
    # self.logger.log_accuracy2noise(accuracy, channel_amplitudes[-1], step)


def first_layer_weights(model):
  for name, weights in model.named_parameters():
    w = weights.abs()
    chn = w.sum(dim=0).sum(-1).sum(-1)
    # Normalize so that R+G+B=1
    chn = chn / chn.sum(0).expand_as(chn)
    chn[torch.isnan(chn)] = 0
    return chn.cpu().detach().numpy()