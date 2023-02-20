import torch.nn as nn
import torch


class UNET(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd9 = nn.BatchNorm2d(1024)
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd10 = nn.BatchNorm2d(1024)
        
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv11 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd12 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd14 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv15 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd15 = nn.BatchNorm2d(128)
        self.conv16 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd16 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv17 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd17 = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd18 = nn.BatchNorm2d(64)

        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)


    def forward(self, x):
        # print(f"input: {x.shape}")
        x1 = self.bnd1(self.relu(self.conv1(x)))
        # print(f"conv1: {x1.shape}")
        x1 = self.bnd2(self.relu(self.conv2(x1)))
        # print(f"conv2: {x1.shape}")
        
        xp = self.pool1(x1)
        # print(f"pool1: {xp.shape}")
        x2 = self.bnd3(self.relu(self.conv3(xp)))
        # print(f"conv3: {x3.shape}")
        x2 = self.bnd4(self.relu(self.conv4(x2)))
        # print(f"conv4: {x4.shape}")
        
        xp = self.pool2(x2)
        # print(f"pool2: {xp.shape}")
        x3 = self.bnd5(self.relu(self.conv5(xp)))
        # print(f"conv5: {x5.shape}")
        x3 = self.bnd6(self.relu(self.conv6(x3)))
        # print(f"conv6: {x6.shape}")
        
        xp = self.pool3(x3)
        # print(f"pool3: {xp.shape}")
        x4 = self.bnd7(self.relu(self.conv7(xp)))
        # print(f"conv7: {x7.shape}")
        x4 = self.bnd8(self.relu(self.conv8(x4)))
        # print(f"conv8: {x8.shape}")

        xp = self.pool4(x4)
        # print(f"pool4: {xp.shape}")
        x5 = self.bnd9(self.relu(self.conv9(xp)))
        # print(f"conv9: {x9.shape}")
        x5 = self.bnd10(self.relu(self.conv10(x5)))
        # print(f"conv10: {x10.shape}")

        xd = self.bnd11(self.relu(self.deconv1(x5)))
        # print(f"deconv1: {xd1.shape}")

        xc = torch.cat((xd,x4), dim=1)
        # print(f"cat1: {xc1.shape}")
        
        x5 = self.bn1(self.relu(self.conv11(xc)))
        # print(f"conv11: {x11.shape}")
        x5 = self.bnd12(self.relu(self.conv12(x5)))
        # print(f"conv12: {x12.shape}")

        xd = self.bn2(self.relu(self.deconv2(x5)))
        # print(f"deconv2: {xd2.shape}")
        xc = torch.cat((xd,x3), dim=1)
        # print(f"cat2: {xc2.shape}")
        x5 = self.bnd13(self.relu(self.conv13(xc)))
        # print(f"conv13: {x13.shape}")
        x5 = self.bnd14(self.relu(self.conv14(x5)))
        # print(f"conv14: {x14.shape}")

        xd = self.bn3(self.relu(self.deconv3(x5)))
        # print(f"deconv3: {xd3.shape}")
        xc = torch.cat((xd,x2), dim=1)
        # print(f"cat3: {xc3.shape}")
        x5 = self.bnd15(self.relu(self.conv15(xc)))
        # print(f"conv15: {x15.shape}")
        x5 = self.bnd16(self.relu(self.conv16(x5)))
        # print(f"conv16: {x16.shape}")

        xd = self.bn4(self.relu(self.deconv4(x5)))
        # print(f"deconv4: {xd4.shape}")
        xc = torch.cat((xd,x1), dim=1)
        # print(f"cat4: {xc4.shape}")
        x5 = self.bnd17(self.relu(self.conv17(xc)))
        # print(f"conv17: {x17.shape}")
        x5 = self.bnd18(self.relu(self.conv18(x5)))
        # print(f"conv18: {x18.shape}")

        score = self.classifier(x5)

        return score  # size=(N, n_class, x.H/1, x.W/1)