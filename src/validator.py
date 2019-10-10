import torch
import torch.nn as nn


class Validator():
  def __init__(self, config, logger):
    self.config = config
    self.logger = logger

  def validate(self, model, val_loader, step):
    def hook_function(module, grad_in, grad_out):
      print(module)
      print(grad_in[1].shape)
      print(grad_out[0].shape)
      qwe

    def printnorm(self, input, output):
      # input is a tuple of packed inputs
      # output is a Tensor. output.data is the Tensor we are interested
      print('Inside ' + self.__class__.__name__ + ' forward')
      print('')
      print('input: ', type(input))
      print('input[0]: ', type(input[0]))
      print('output: ', type(output))
      print('')
      print('input size:', input[0].size())
      print('output size:', output.data.size())
      print('output norm:', output.data.norm())

    def printgradnorm(self, grad_input, grad_output):
      print('Inside ' + self.__class__.__name__ + ' backward')
      print('Inside class:' + self.__class__.__name__)
      print('')
      print('grad_input: ', type(grad_input))
      print('grad_input[0]: ', type(grad_input[0]))
      print('grad_output: ', type(grad_output))
      print('grad_output[0]: ', type(grad_output[0]))
      print('')
      print('grad_input size:', grad_input[0].size())
      print('grad_output size:', grad_output[0].size())
      print('grad_input norm:', grad_input[0].norm())

    first_layer = model.backbone.conv1
    # first_layer = list(model.features._modules.items())[0][1]
    print(first_layer)
    first_layer.register_forward_hook(printnorm)
    first_layer.register_backward_hook(printgradnorm)
    # first_layer.register_backward_hook(hook_function)

    corrects = torch.tensor([], dtype=bool)
    for batch_i, data in enumerate(val_loader, 1):
      if batch_i > self.config.max_val_batches:
        print("WOOOO")
        break

      inputs, labels = data
      print(inputs.shape)
      print(inputs.requires_grad)
      # Put requires gradients = True for input image
      qwe
      inputs = torch.randn(self.config.batch_size, 3, 100, 100)

      outputs = model(inputs)
      model.zero_grad()
      loss, accuracy = model.calc_loss(outputs, labels, accuracy=True)
      loss.backward()
      # preds = model.predict(inputs)
      # corrects = torch.cat((corrects, preds.cpu() == labels))

    accuracy = corrects.sum(dtype=float) / len(corrects)
    self.logger.log_accuracy(accuracy, step)
    channel_amplitudes = first_layer_weights(model)
    self.logger.log_first_layer(channel_amplitudes, step)
    # self.logger.log_accuracy2noise(accuracy, channel_amplitudes[-1], step)
    qwe


def first_layer_weights(model):
  for name, weights in model.named_parameters():
    w = weights.abs()
    chn = w.sum(dim=0).sum(-1).sum(-1)
    # Normalize so that R+G+B=1
    chn = chn / chn.sum(0).expand_as(chn)
    chn[torch.isnan(chn)] = 0
    return chn.cpu().detach().numpy()
