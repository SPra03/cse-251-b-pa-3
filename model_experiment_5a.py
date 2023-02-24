import torch.nn as nn
import torch
    
class Model_5A(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5 = nn.BatchNorm2d(512)
        
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd6 = nn.BatchNorm2d(1024)        
        self.conv7 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd7 = nn.BatchNorm2d(2048)       
        
        self.relu = nn.ReLU(inplace=True)
        
        # decoder
        
        #x8
        self.deconv1 = nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0)
        self.bn1 = nn.BatchNorm2d(2048)
        
        #x9
        self.deconv2 = nn.ConvTranspose2d(2048+1024, 1024, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0)
        self.bn2 = nn.BatchNorm2d(1024)
        self.deconv3 = nn.ConvTranspose2d(1024+512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.deconv4 = nn.ConvTranspose2d(512+256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.deconv5 = nn.ConvTranspose2d(256+128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.deconv6 = nn.ConvTranspose2d(128+64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.deconv7 = nn.ConvTranspose2d(64+32, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        """
        x: torch.Size([16, 3, 224, 224]), x1: torch.Size([16, 32, 112, 112]), x2: torch.Size([16, 64, 56, 56]), x3: torch.Size([16, 128, 28, 28]), x4: torch.Size([16, 256, 14, 14]), x5: torch.Size([16, 512, 7, 7]), x6: torch.Size([16, 512, 14, 14]), x7:torch.Size([16, 256, 28, 28]), x8: torch.Size([16, 128, 56, 56]), x9: torch.Size([16, 64, 112, 112]), x10: torch.Size([16,32, 224, 224])"""
        x1 = self.bnd1(self.relu(self.conv1(x)))
        x2 = self.bnd2(self.relu(self.conv2(x1)))
        x3 = self.bnd3(self.relu(self.conv3(x2)))
        x4 = self.bnd4(self.relu(self.conv4(x3)))
        x5 = self.bnd5(self.relu(self.conv5(x4)))
        
        x6 = self.bnd6(self.relu(self.conv6(x5)))
        x7 = self.bnd7(self.relu(self.conv7(x6)))
        
        #### Decoder 
        
        x8 = self.bn1(self.relu(self.deconv1(x7)))    
        
#         print(f" x: {x.shape},\n x1: {x1.shape},\n x2: {x2.shape},\n x3: {x3.shape}, \n x4: {x4.shape}, \n x5: {x5.shape}, \n x6: {x6.shape}, \n x7: {x7.shape}, \n x8: {x8.shape}")#, x9: {x9.shape}")#, x10: {x10.shape}")
        
        x9 = self.bn2(self.relu(self.deconv2(torch.cat([x6, x8], dim=1))))
        
#         print(f" x: {x.shape},\n x1: {x1.shape},\n x2: {x2.shape},\n x3: {x3.shape}, \n x4: {x4.shape}, \n x5: {x5.shape}, \n x6: {x6.shape}, \n x7: {x7.shape}, \n x8: {x8.shape}, x9: {x9.shape}")#, x10: {x10.shape}")
            
        x10 = self.bn3(self.relu(self.deconv3(torch.cat([x5, x9], dim=1))))
        x11 = self.bn4(self.relu(self.deconv4(torch.cat([x4, x10], dim=1))))
        x12 = self.bn5(self.relu(self.deconv5(torch.cat([x3, x11], dim=1))))
        x13 = self.bn6(self.relu(self.deconv6(torch.cat([x2, x12], dim=1))))
        x14 = self.bn7(self.relu(self.deconv7(torch.cat([x1, x13], dim=1))))

        score = self.classifier(x14)
        
        return score  # size=(N, n_class, x.H/1, x.W/1)
