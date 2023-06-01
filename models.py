import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
from einops import rearrange
from einops.layers.torch import Rearrange

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
def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class A5(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, block_types=['C', 'T', 'C', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], 1, bias=False)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

#-------------------------------------------------------------------
# A4 model with linear layers replaced with transformers.
class A6(nn.Module):
    def __init__(self):
        super(A6,self).__init__()

        # image part
        self.cnn = nn.Sequential(
        nn.Conv2d(5,32,3,padding=1),
        nn.PReLU(),
        nn.AvgPool2d(2,2),
        Inception_block(32,16),
        Transformer(64, 64, (16,16)),
        Inception_block(64,16),
        Transformer(64, 64, (16,16)),
        Inception_block(64,16),
        nn.AvgPool2d(2,2),
        Inception_block(64,8),
        Transformer(32, 32, (8,8)),
        Inception_block(32,8),
        Transformer(32, 32, (8,8)),
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
