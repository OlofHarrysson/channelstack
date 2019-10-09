import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path


def get_model(config):
  model = MyModel(config)
  return model.to(model.device)


class MyModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cuda' if config.use_gpu else 'cpu'
    self.loss_fn = nn.CrossEntropyLoss()

    backbone = models.resnet18(pretrained=config.pretrained)
    backbone.conv1 = nn.Conv2d(config.im_channels,
                               backbone.conv1.out_channels,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
    backbone.fc = nn.Linear(backbone.fc.in_features, 10)

    self.backbone = backbone

  def forward(self, inputs):
    inputs = inputs.to(self.device)
    return self.backbone(inputs)

  def predict(self, inputs):
    with torch.no_grad():
      outputs = self(inputs)
      _, preds = torch.max(outputs, 1)
      return preds

  def calc_loss(self, outputs, labels, accuracy=False):
    labels = labels.to(self.device)

    if accuracy:
      _, preds = torch.max(outputs, 1)
      accuracy = torch.sum(preds == labels, dtype=float) / len(preds)
      return self.loss_fn(outputs, labels), accuracy
    return self.loss_fn(outputs, labels)

  def save(self, path):
    save_dir = Path(path).parent
    save_dir.mkdir(exist_ok=True, parents=True)
    print("Saving Weights @ " + path)
    torch.save(self.state_dict(), path)

  def load(self, path):
    print('Loading weights from {}'.format(path))
    weights = torch.load(path, map_location='cpu')
    self.load_state_dict(weights, strict=False)