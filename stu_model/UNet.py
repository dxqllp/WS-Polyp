import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self, num_classes=1):
        super(Unet, self).__init__()
        chs = [64, 128, 256, 512, 1024]
        # Conv block 1 - Down 1
        self.conv11 = nn.Sequential(nn.Conv2d(3, chs[0], 3, padding=1, stride=1),
                                    nn.BatchNorm2d(chs[0]),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(chs[0], chs[0], 3, padding=1, stride=1),
                                    nn.BatchNorm2d(chs[0]),
                                    nn.ReLU(inplace=True))
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2 - Down 2
        self.conv21 = nn.Sequential(nn.Conv2d(chs[0], chs[1], 3, padding=1, stride=1),
                                    nn.BatchNorm2d(chs[1]),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(chs[1], chs[1], 3, padding=1, stride=1),
                                    nn.BatchNorm2d(chs[1]),
                                    nn.ReLU(inplace=True))
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3 - Down 3
        self.conv31 = nn.Sequential(nn.Conv2d(chs[1], chs[2], 3, padding=1, stride=1),
                                    nn.BatchNorm2d(chs[2]),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(chs[2], chs[2], 3, padding=1, stride=1),
                                    nn.BatchNorm2d(chs[2]),
                                    nn.ReLU(inplace=True))
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 4 - Down 4
        self.conv41 = nn.Sequential(nn.Conv2d(chs[2], chs[3], 3, padding=1, stride=1),
                                    nn.BatchNorm2d(chs[3]),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(chs[3], chs[3], 3, padding=1, stride=1),
                                    nn.BatchNorm2d(chs[3]),
                                    nn.ReLU(inplace=True))
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(nn.Conv2d(chs[3], chs[4], 3, padding=1, stride=1),
                                   nn.BatchNorm2d(chs[4]),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(chs[4], chs[4], 3, padding=1, stride=1),
                                   nn.BatchNorm2d(chs[4]),
                                   nn.ReLU(inplace=True))

        # Up 1
        self.up_1 = nn.ConvTranspose2d(chs[4], chs[3], kernel_size=2, stride=2)
        self.conv_up_1 = nn.Sequential(nn.Conv2d(chs[4], chs[3], 3, padding=1, stride=1),
                                       nn.BatchNorm2d(chs[3]),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(chs[3], chs[3], 3, padding=1, stride=1),
                                       nn.ReLU(inplace=True))

        # Up 2
        self.up_2 = nn.ConvTranspose2d(chs[3], chs[2], kernel_size=2, stride=2)
        self.conv_up_2 = nn.Sequential(nn.Conv2d(chs[3], chs[2], 3, padding=1, stride=1),
                                       nn.BatchNorm2d(chs[2]),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(chs[2], chs[2], 3, padding=1, stride=1),
                                       nn.ReLU(inplace=True))

        # Up 3
        self.up_3 = nn.ConvTranspose2d(chs[2], chs[1], kernel_size=2, stride=2)
        self.conv_up_3 = nn.Sequential(nn.Conv2d(chs[2], chs[1], 3, padding=1, stride=1),
                                       nn.BatchNorm2d(chs[1]),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(chs[1], chs[1], 3, padding=1, stride=1),
                                       nn.ReLU(inplace=True))
        # Up 4
        self.up_4 = nn.ConvTranspose2d(chs[1], chs[0], kernel_size=2, stride=2)
        self.conv_up_4 = nn.Sequential(nn.Conv2d(chs[1], chs[0], 3, padding=1, stride=1),
                                       nn.BatchNorm2d(chs[0]),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(chs[0], chs[0], 3, padding=1, stride=1),
                                       nn.ReLU(inplace=True))
        # Final output
        # self.predict = nn.Conv2d(chs[3], 1, kernel_size=1)
        self.conv_final = nn.Conv2d(chs[0], out_channels=num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        # Down 1
        x11 = self.conv11(x)
        x12 = self.max1(x11)

        # Down 2
        x21 = self.conv21(x12)
        x22 = self.max2(x21)

        # Down 3
        x31 = self.conv31(x22)
        x32 = self.max3(x31)

        x41 = self.conv41(x32)
        x42 = self.max4(x41)

        x5 = self.conv5(x42)

        # Up 1
        x = self.up_1(x5)
        x = torch.cat((x, x41), dim=1)
        y1 = self.conv_up_1(x)

        # Up 2
        x = self.up_2(y1)
        x = torch.cat((x, x31), dim=1)
        y2 = self.conv_up_2(x)

        # Up 3
        x = self.up_3(y2)
        x = torch.cat((x, x21), dim=1)
        y3 = self.conv_up_3(x)

        # Up 4
        x = self.up_4(y3)
        x = torch.cat((x, x11), dim=1)
        y4 = self.conv_up_4(x)

        # Final output
        out = self.conv_final(y4)

        return torch.sigmoid(out)


if __name__ == '__main__':
    from thop import profile, clever_format
    model = Unet(num_classes=1)
    input = torch.randn([1, 3, 352, 352])
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))