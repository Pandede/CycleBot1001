import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.prefix_layers = nn.Sequential(self.conv_block(3, 16),
                                           self.conv_block(16, 32),
                                           self.conv_block(32, 64))
        self.core_layers = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        self.postfix_layers = nn.Sequential(self.conv_block(64, 32),
                                            self.conv_block(32, 16),
                                            self.conv_block(16, 8))
        self.output_layer = nn.Conv2d(8, 3, 1)

    @staticmethod
    def conv_block(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(nn.ReflectionPad2d(1),
                             nn.Conv2d(in_channels, out_channels, kernel_size=3),
                             nn.BatchNorm2d(out_channels, momentum=.8),
                             nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prefix = self.prefix_layers(x)
        core = self.core_layers(prefix)
        postfix = self.postfix_layers(core)
        return torch.tanh(self.output_layer(postfix))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.res_block = Generator.conv_block(channels, channels)

    def forward(self, x) -> torch.Tensor:
        return x + self.res_block(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = self.__conv_block(3, 128)
        self.conv2 = self.__conv_block(128, 64)
        self.conv3 = self.__conv_block(64, 32)
        self.conv4 = self.__conv_block(32, 16)
        self.conv5 = self.__conv_block(16, 8)
        self.output = nn.Linear(8 * 7 * 7, 1)

    @staticmethod
    def __conv_block(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=2),
                             nn.LeakyReLU(.2, inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = torch.flatten(h, 1)
        return self.output(h)
