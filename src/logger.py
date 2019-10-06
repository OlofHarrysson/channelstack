import visdom
import pandas as pd
import numpy as np


def clear_envs(viz):
  [viz.close(env=env) for env in viz.get_env_list()]  # Kills wind
  # [viz.delete_env(env) for env in viz.get_env_list()] # Kills envs


class EMAverage(object):
  ''' Smooths the curve with Exponential Moving Average '''
  def __init__(self, time_steps):
    self.vals = deque([], time_steps)

  def update(self, val):
    self.vals.append(val)
    df = pd.Series(self.vals)
    return df.ewm(com=0.5).mean().mean()


class Logger():
  def __init__(self, config):
    self.config = config
    self.viz = visdom.Visdom(port='6006')
    clear_envs(self.viz)

  def log_accuracy(self, acc, step):
    self.viz.line(
      Y=[acc],
      X=[step],
      update='append',
      win='Validation Accuracy',
      opts=dict(
        xlabel='Steps',
        ylabel='Accuracy',
        title='Validation Accuracy',
        # ytickmin = 0,
        # ytickmax = 1,
      ))

  def log_first_layer(self, channel_amplitudes, step):
    legend = ['Red', 'Green', 'Blue', 'Noise']
    self.viz.line(Y=channel_amplitudes.reshape(1, -1),
                  X=[step],
                  update='append',
                  win='Channel Amplitudes',
                  opts=dict(
                    xlabel='Steps',
                    ylabel='Accuracy',
                    title='Channel Amplitudes',
                  ))

  def log_accuracy2noise(self, acc, noise, step):
    Y = np.array([acc, noise]).reshape((1, -1))
    self.viz.line(Y=Y,
                  X=[step],
                  update='append',
                  win='Accuracy vs Noise Importance',
                  opts=dict(xlabel='Steps',
                            ylabel='Accuracy / Importance ',
                            title='Accuracy vs Noise Importance',
                            legend=['Accuracy', 'Noise']))

  def log_channels(self, channels, names=None):
    print(channels.shape)
    for i, channel in enumerate(channels, 1):
      caption = i
      self.viz.image(channel, opts=dict(caption=caption))
