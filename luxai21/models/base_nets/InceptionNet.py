import torch
import torch.nn as nn


class InceptionNet_v1(nn.Module):
    """
    Inspired by GoogLeNet (https://arxiv.org/pdf/1409.4842.pdf)
    Current architecture:
        ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 16, 32, 32]           7,216
           BatchNorm2d-2           [-1, 16, 32, 32]              32
                  ReLU-3           [-1, 16, 32, 32]               0
            conv_block-4           [-1, 16, 32, 32]               0
             MaxPool2d-5           [-1, 16, 16, 16]               0
                Conv2d-6           [-1, 32, 16, 16]           4,640
           BatchNorm2d-7           [-1, 32, 16, 16]              64
                  ReLU-8           [-1, 32, 16, 16]               0
            conv_block-9           [-1, 32, 16, 16]               0
            MaxPool2d-10             [-1, 32, 8, 8]               0
               Conv2d-11             [-1, 32, 8, 8]           1,056
          BatchNorm2d-12             [-1, 32, 8, 8]              64
                 ReLU-13             [-1, 32, 8, 8]               0
           conv_block-14             [-1, 32, 8, 8]               0
               Conv2d-15             [-1, 96, 8, 8]           3,168
          BatchNorm2d-16             [-1, 96, 8, 8]             192
                 ReLU-17             [-1, 96, 8, 8]               0
           conv_block-18             [-1, 96, 8, 8]               0
               Conv2d-19             [-1, 64, 8, 8]          55,360
          BatchNorm2d-20             [-1, 64, 8, 8]             128
                 ReLU-21             [-1, 64, 8, 8]               0
           conv_block-22             [-1, 64, 8, 8]               0
               Conv2d-23             [-1, 16, 8, 8]             528
          BatchNorm2d-24             [-1, 16, 8, 8]              32
                 ReLU-25             [-1, 16, 8, 8]               0
           conv_block-26             [-1, 16, 8, 8]               0
               Conv2d-27             [-1, 16, 8, 8]           6,416
          BatchNorm2d-28             [-1, 16, 8, 8]              32
                 ReLU-29             [-1, 16, 8, 8]               0
           conv_block-30             [-1, 16, 8, 8]               0
            MaxPool2d-31             [-1, 32, 8, 8]               0
               Conv2d-32             [-1, 32, 8, 8]           1,056
          BatchNorm2d-33             [-1, 32, 8, 8]              64
                 ReLU-34             [-1, 32, 8, 8]               0
           conv_block-35             [-1, 32, 8, 8]               0
      Inception_Block-36            [-1, 144, 8, 8]               0
            MaxPool2d-37            [-1, 144, 4, 4]               0
               Conv2d-38             [-1, 96, 4, 4]          13,920
          BatchNorm2d-39             [-1, 96, 4, 4]             192
                 ReLU-40             [-1, 96, 4, 4]               0
           conv_block-41             [-1, 96, 4, 4]               0
               Conv2d-42             [-1, 96, 4, 4]          13,920
          BatchNorm2d-43             [-1, 96, 4, 4]             192
                 ReLU-44             [-1, 96, 4, 4]               0
           conv_block-45             [-1, 96, 4, 4]               0
               Conv2d-46            [-1, 104, 4, 4]          89,960
          BatchNorm2d-47            [-1, 104, 4, 4]             208
                 ReLU-48            [-1, 104, 4, 4]               0
           conv_block-49            [-1, 104, 4, 4]               0
               Conv2d-50             [-1, 16, 4, 4]           2,320
          BatchNorm2d-51             [-1, 16, 4, 4]              32
                 ReLU-52             [-1, 16, 4, 4]               0
           conv_block-53             [-1, 16, 4, 4]               0
               Conv2d-54             [-1, 24, 4, 4]           9,624
          BatchNorm2d-55             [-1, 24, 4, 4]              48
                 ReLU-56             [-1, 24, 4, 4]               0
           conv_block-57             [-1, 24, 4, 4]               0
            MaxPool2d-58            [-1, 144, 4, 4]               0
               Conv2d-59             [-1, 64, 4, 4]           9,280
          BatchNorm2d-60             [-1, 64, 4, 4]             128
                 ReLU-61             [-1, 64, 4, 4]               0
           conv_block-62             [-1, 64, 4, 4]               0
      Inception_Block-63            [-1, 288, 4, 4]               0
            AvgPool2d-64            [-1, 288, 3, 3]               0
    ================================================================
    Total params: 219,872
    Trainable params: 219,872
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.07
    Forward/backward pass size (MB): 1.69
    Params size (MB): 0.84
    Estimated Total Size (MB): 2.60
    ----------------------------------------------------------------
    """

    def __init__(self, in_channels):
        super(InceptionNet_v1, self).__init__()
        self.conv1 = conv_block(in_channels=in_channels, out_channels=16, kernel_size=(5, 5), stride=(1, 1),
                                padding=(2, 2))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))

        # Inception block
        self.inception3 = Inception_Block(in_channels=32, out_1x1x1=32, red_3x3x3=96, out_3x3x3=64, red_5x5x5=16,
                                          out_5x5x5=16, out_1x1x1_pool=32)
        self.inception4 = Inception_Block(in_channels=144, out_1x1x1=96, red_3x3x3=96, out_3x3x3=104, red_5x5x5=16,
                                          out_5x5x5=24, out_1x1x1_pool=64)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)

        self.output_size = 2592

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.inception3(x)
        x = self.maxpool(x)
        x = self.inception4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        return x


class InceptionNet_v2(nn.Module):
    """
    Inspired by GoogLeNet (https://arxiv.org/pdf/1409.4842.pdf)
    Note: Depthwise Convolution in the first two layers
    Current architecture:
        ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 18, 32, 32]             468
           BatchNorm2d-2           [-1, 18, 32, 32]              36
                  ReLU-3           [-1, 18, 32, 32]               0
            conv_block-4           [-1, 18, 32, 32]               0
             MaxPool2d-5           [-1, 18, 16, 16]               0
                Conv2d-6           [-1, 36, 16, 16]             360
           BatchNorm2d-7           [-1, 36, 16, 16]              72
                  ReLU-8           [-1, 36, 16, 16]               0
            conv_block-9           [-1, 36, 16, 16]               0
            MaxPool2d-10             [-1, 36, 8, 8]               0
               Conv2d-11              [-1, 8, 8, 8]             296
          BatchNorm2d-12              [-1, 8, 8, 8]              16
                 ReLU-13              [-1, 8, 8, 8]               0
           conv_block-14              [-1, 8, 8, 8]               0
               Conv2d-15             [-1, 96, 8, 8]           3,552
          BatchNorm2d-16             [-1, 96, 8, 8]             192
                 ReLU-17             [-1, 96, 8, 8]               0
           conv_block-18             [-1, 96, 8, 8]               0
               Conv2d-19             [-1, 16, 8, 8]          13,840
          BatchNorm2d-20             [-1, 16, 8, 8]              32
                 ReLU-21             [-1, 16, 8, 8]               0
           conv_block-22             [-1, 16, 8, 8]               0
               Conv2d-23             [-1, 16, 8, 8]             592
          BatchNorm2d-24             [-1, 16, 8, 8]              32
                 ReLU-25             [-1, 16, 8, 8]               0
           conv_block-26             [-1, 16, 8, 8]               0
               Conv2d-27              [-1, 8, 8, 8]           3,208
          BatchNorm2d-28              [-1, 8, 8, 8]              16
                 ReLU-29              [-1, 8, 8, 8]               0
           conv_block-30              [-1, 8, 8, 8]               0
            MaxPool2d-31             [-1, 36, 8, 8]               0
               Conv2d-32             [-1, 16, 8, 8]             592
          BatchNorm2d-33             [-1, 16, 8, 8]              32
                 ReLU-34             [-1, 16, 8, 8]               0
           conv_block-35             [-1, 16, 8, 8]               0
      Inception_Block-36             [-1, 48, 8, 8]               0
            MaxPool2d-37             [-1, 48, 4, 4]               0
               Conv2d-38             [-1, 16, 4, 4]             784
          BatchNorm2d-39             [-1, 16, 4, 4]              32
                 ReLU-40             [-1, 16, 4, 4]               0
           conv_block-41             [-1, 16, 4, 4]               0
               Conv2d-42             [-1, 96, 4, 4]           4,704
          BatchNorm2d-43             [-1, 96, 4, 4]             192
                 ReLU-44             [-1, 96, 4, 4]               0
           conv_block-45             [-1, 96, 4, 4]               0
               Conv2d-46             [-1, 32, 4, 4]          27,680
          BatchNorm2d-47             [-1, 32, 4, 4]              64
                 ReLU-48             [-1, 32, 4, 4]               0
           conv_block-49             [-1, 32, 4, 4]               0
               Conv2d-50             [-1, 16, 4, 4]             784
          BatchNorm2d-51             [-1, 16, 4, 4]              32
                 ReLU-52             [-1, 16, 4, 4]               0
           conv_block-53             [-1, 16, 4, 4]               0
               Conv2d-54             [-1, 16, 4, 4]           6,416
          BatchNorm2d-55             [-1, 16, 4, 4]              32
                 ReLU-56             [-1, 16, 4, 4]               0
           conv_block-57             [-1, 16, 4, 4]               0
            MaxPool2d-58             [-1, 48, 4, 4]               0
               Conv2d-59             [-1, 32, 4, 4]           1,568
          BatchNorm2d-60             [-1, 32, 4, 4]              64
                 ReLU-61             [-1, 32, 4, 4]               0
           conv_block-62             [-1, 32, 4, 4]               0
      Inception_Block-63             [-1, 96, 4, 4]               0
            AvgPool2d-64             [-1, 96, 3, 3]               0
    ================================================================
    Total params: 65,688
    Trainable params: 65,688
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.07
    Forward/backward pass size (MB): 1.38
    Params size (MB): 0.25
    Estimated Total Size (MB): 1.70
    ----------------------------------------------------------------
    """

    def __init__(self, in_channels):
        super(InceptionNet_v2, self).__init__()
        self.conv1 = conv_block(in_channels=in_channels, out_channels=in_channels, kernel_size=(5, 5), stride=(1, 1),
                                padding=(2, 2), groups=in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv_block(in_channels=in_channels, out_channels=in_channels*2, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1), groups=in_channels)

        # Inception block
        self.inception3 = Inception_Block(in_channels=in_channels*2, out_1x1x1=8, red_3x3x3=96, out_3x3x3=16, red_5x5x5=16,
                                          out_5x5x5=8, out_1x1x1_pool=16)
        self.inception4 = Inception_Block(in_channels=48, out_1x1x1=16, red_3x3x3=96, out_3x3x3=32, red_5x5x5=16,
                                          out_5x5x5=16, out_1x1x1_pool=32)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)

        self.output_size = 864

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.inception3(x)
        x = self.maxpool(x)
        x = self.inception4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        return x


class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_1x1x1, red_3x3x3, out_3x3x3, red_5x5x5, out_5x5x5, out_1x1x1_pool):
        super(Inception_Block, self).__init__()

        self.branch_1 = conv_block(in_channels, out_1x1x1, kernel_size=1)
        self.branch_2 = nn.Sequential(conv_block(in_channels, red_3x3x3, kernel_size=1),
                                      conv_block(red_3x3x3, out_3x3x3, kernel_size=3, stride=1, padding=1))
        self.branch_3 = nn.Sequential(conv_block(in_channels, red_5x5x5, kernel_size=1),
                                      conv_block(red_5x5x5, out_5x5x5, kernel_size=5, stride=1, padding=2))
        self.branch_4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                      conv_block(in_channels, out_1x1x1_pool, kernel_size=1))

    def forward(self, x):
        return torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)], 1)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))