import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = self.__conv_block(3, 256)
        self.conv2 = self.__conv_block(256, 128)
        self.conv3 = self.__conv_block(128, 64)
        self.conv4 = self.__conv_block(64, 32)
        self.conv5 = self.__conv_block(32, 3)

    @staticmethod
    def __conv_block(in_channels: int, out_channels: int):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3),
                             nn.ReflectionPad2d(1),
                             nn.BatchNorm2d(out_channels, momentum=.8),
                             nn.LeakyReLU(.2, inplace=True))

    def forward(self, x: torch.Tensor):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        return torch.tanh(self.conv5(h))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = self.__conv_block(3, 64)
        self.conv2 = self.__conv_block(64, 32)
        self.conv3 = self.__conv_block(32, 16)
        self.conv4 = self.__conv_block(16, 8)
        self.conv5 = self.__conv_block(8, 4)
        self.output = nn.Linear(4 * 7 * 7, 1)

    @staticmethod
    def __conv_block(in_channels: int, out_channels: int):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=2),
                             nn.LeakyReLU(.2, inplace=True))

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = torch.flatten(h, 1)
        return self.output(h)
