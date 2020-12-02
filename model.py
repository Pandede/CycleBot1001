import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.head_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                        nn.Conv2d(3, 32, kernel_size=7),
                                        nn.BatchNorm2d(32, momentum=.8),
                                        nn.ReLU(inplace=True))
        self.prefix_layers = nn.Sequential(self.conv_block(32, 64),
                                           self.conv_block(64, 128))
        self.core_layers = nn.Sequential(*[ResidualBlock(128) for _ in range(6)])
        self.postfix_layers = nn.Sequential(self.deconv_block(128, 64),
                                            self.deconv_block(64, 32))
        self.output_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                          nn.Conv2d(32, 3, kernel_size=7),
                                          nn.Tanh())

    @staticmethod
    def conv_block(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(nn.ReflectionPad2d(1),
                             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False),
                             nn.BatchNorm2d(out_channels, momentum=.8),
                             nn.ReLU(inplace=True))

    @staticmethod
    def deconv_block(in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                             nn.Upsample(scale_factor=2),
                             nn.BatchNorm2d(out_channels, momentum=.8),
                             nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head_layer(x)
        prefix = self.prefix_layers(head)
        core = self.core_layers(prefix)
        postfix = self.postfix_layers(core)
        return self.output_layer(postfix)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.res_block = self.block(channels)

    @staticmethod
    def block(channels: int):
        return nn.Sequential(nn.ReflectionPad2d(1),
                             nn.Conv2d(channels, channels, kernel_size=3, bias=False),
                             nn.BatchNorm2d(channels, momentum=.8),
                             nn.ReLU(inplace=True),
                             nn.ReflectionPad2d(1),
                             nn.Conv2d(channels, channels, kernel_size=3, bias=False),
                             nn.BatchNorm2d(channels, momentum=.8))

    def forward(self, x) -> torch.Tensor:
        return x + self.res_block(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = self.__conv_block(3, 256)
        self.conv2 = self.__conv_block(256, 128)
        self.conv3 = self.__conv_block(128, 64)
        self.conv4 = self.__conv_block(64, 32)
        self.conv5 = self.__conv_block(32, 16)
        self.output = nn.Linear(16 * 7 * 7, 1)

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
