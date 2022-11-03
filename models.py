import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable

#----------------------------------------------------------------
class A1(nn.Module):
    def __init__(self):
        super(A1,self).__init__()
        self.conv1 = nn.Conv2d(5,32,3,padding=1)
        self.pool1 = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.pool2 = nn.AvgPool2d(2,2)
        #self.conv3 = nn.Conv2d(64,128,3,padding=1)
        #self.pool3 = nn.AvgPool2d(2,2)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 32)
        self.fc3 = nn.Linear(32, 1)
        self.p1 = nn.PReLU()
        self.p2 = nn.PReLU()
        self.p3 = nn.PReLU()
        self.p4 = nn.PReLU()

    def forward(self, x):
        x = self.pool1(self.p1(self.conv1(x)))
        x = self.pool2(self.p2(self.conv2(x)))
        #x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.p3(self.fc1(x))
        x = self.p4(self.fc2(x))
        x = self.fc3(x)
        return x
#----------------------------------------------------------------
class A2(nn.Module):
    def __init__(self):
        super(A2,self).__init__()
        # Image parser.
        self.conv1 = nn.Conv2d(5,32,3,padding=1)
        self.pool1 = nn.AvgPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.pool2 = nn.AvgPool2d(2,2)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 32)

        #extra features.
        self.fc3 = nn.Linear(5, 1024)
        self.fc4 = nn.Linear(1024,1024)
        self.fc5 = nn.Linear(1024,1024)
        self.fc6 = nn.Linear(1024,1024)
        self.fc7 = nn.Linear(1024,1024)

        #final part
        self.fc8 = nn.Linear(1056,1024)
        self.fc9 = nn.Linear(1024,1)
        self.p1 = nn.PReLU()
        self.p2 = nn.PReLU()
        self.p3 = nn.PReLU()
        self.p4 = nn.PReLU()

        self.p5 = nn.PReLU()
        self.p6 = nn.PReLU()
        self.p7 = nn.PReLU()
        self.p8 = nn.PReLU()
        self.p9 = nn.PReLU()
        self.p10 = nn.PReLU()
        self.p11 = nn.PReLU()


    def forward(self, x,w):
        # image part
        #w = u[1]
        #x = u[0]
        x = self.pool1(self.p1(self.conv1(x)))
        x = self.pool2(self.p2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.p3(self.fc1(x))
        x = self.p4(self.fc2(x))

        # extra features
        w = self.p5(self.fc3(w))
        w = self.p6(self.fc4(w))
        w = self.p7(self.fc5(w))
        w = self.p8(self.fc6(w))
        w = self.p9(self.fc7(w))

        u = torch.cat((x,w),1)
        u = self.p10(self.fc8(u))
        u = self.p11(self.fc9(u))
        return u
#-------------------------------------------------------------------
class Inception_block(nn.Module):
    def __init__(self,in_channel,out_channel,clip=False):
        super(Inception_block, self).__init__()
        self.clip = clip
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel , kernel_size=1),
            nn.PReLU()
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel , kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(out_channel,out_channel , kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel,out_channel , kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(out_channel,out_channel, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(1,1, count_include_pad=False),
            nn.Conv2d(in_channel,out_channel , kernel_size=1, stride=1),
            nn.PReLU()
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x3 = self.branch3(x)
        if self.clip is False:
            x2 = self.branch2(x)
            return torch.cat((x0,x1,x2,x3),1)
        return torch.cat((x0, x1, x3), 1)

#-------------------------------------------------------------------
class A3(nn.Module):
    def __init__(self):
        super(A3,self).__init__()
        self.conv1 = nn.Conv2d(5,32,3,padding=1)
        self.pool1 = nn.AvgPool2d(2,2)
        self.I1 = Inception_block(32,16)
        self.I2 = Inception_block(64,16)
        self.pool2 = nn.AvgPool2d(2,2)
        self.I3 = Inception_block(64,8)
        self.I4 = Inception_block(32,8)
        self.pool3 = nn.AvgPool2d(2,2)
        self.I5 = Inception_block(32,4,True)
        self.fc1 = nn.Linear(192,1096)
        self.fc2 = nn.Linear(1096,1096)
        self.fc3 = nn.Linear(1096,1)
        self.p1 = nn.PReLU()
        self.p2 = nn.PReLU()
        self.p3 = nn.PReLU()
        self.drop_layer = nn.Dropout(inplace=True)

    def forward(self, x):
        x = self.pool1(self.p1(self.conv1(x)))
        x = self.pool2(self.I2(self.I1(x)))
        x = self.pool3(self.I4(self.I3(x)))
        x = self.I5(x)
        x = torch.flatten(x, 1)
        x = self.p2(self.drop_layer(self.fc1(x)))
        x = self.p3(self.drop_layer(self.fc2(x)))
        x = self.fc3(x)
        return x

#-------------------------------------------------------------------
#-------------------------------------------------------------------
class A4(nn.Module):
    def __init__(self):
        super(A4,self).__init__()

        # image part
        self.cnn = nn.Sequential(
        nn.Conv2d(5,32,3,padding=1),
        nn.PReLU(),
        nn.AvgPool2d(2,2),
        Inception_block(32,16),
        Inception_block(64,16),
        Inception_block(64,16),
        nn.AvgPool2d(2,2),
        Inception_block(64,8),
        Inception_block(32,8),
        Inception_block(32,8),
        nn.AvgPool2d(2,2),
        Inception_block(32,4,True)
        )

        self.cnn2 = nn.Sequential(
        nn.Linear(192,1096),
        nn.Dropout(),
        nn.PReLU(),
        nn.Linear(1096,1096),
        nn.Dropout(),
        nn.PReLU()
        )

        # extra features
        self.mixinp = nn.Sequential(
        nn.Linear(5, 1024),
        nn.Dropout(),
        nn.PReLU(),
        nn.Linear(1024,512),
        nn.Dropout(),
        nn.PReLU(),
        nn.Linear(512,512),
        nn.Dropout(),
        nn.PReLU(),
        nn.Linear(512,512),
        nn.Dropout(),
        nn.PReLU(),
        nn.Linear(512,1024),
        nn.Dropout(),
        nn.PReLU()
        )

        #final part
        self.final = nn.Sequential(
        nn.Linear(2120,1024),
        nn.Dropout(),
        nn.PReLU(),
        nn.Linear(1024,1)
        )



    def forward(self, x, w):
        # combine both streams.
        x = self.cnn(x)
        x = torch.flatten(x,1)
        x = self.cnn2(x)
        w = self.mixinp(w)
        u = torch.cat((x,w),1)
        u = self.final(u)
        return u

#-------------------------------------------------------------------
# Future Work - use vision transformers with CNN
#-------------------------------------------------------------------
class A5(nn.Module):
    def __init__(self):
        super(A5,self).__init__()

    def forward(self, x):

        return x

#-------------------------------------------------------------------
