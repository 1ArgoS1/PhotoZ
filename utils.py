import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------
# NMAD as a loss function.
def Metrics(prediction,z):
    err_abs = np.sum(np.abs(prediction - z)) / z.shape[0]
    deltaz = (prediction - z) / (1 + z)
    bias = np.sum(deltaz) / z.shape[0]
    nmad = 1.48 * np.median(np.abs(deltaz - np.median(deltaz)))
    f_outlier = (np.count_nonzero(np.abs(deltaz) >= 0.05))/z.shape[0]*100
    return {"nmad":nmad, "bias":bias, "MAE":err_abs, "frac":f_outlier, 'zspec':prediction, "zphoto":z}

#------------------------------------------------------------------------------
def format_(data,filename="temp"):
    # Evaluate the performance metrics.
    avg_nmad = 0.0
    avg_bias = 0.0
    avg_MAE = 0.0
    avg_loss = 0.0
    avg_frac = 0
    z1 = []
    z2 = []
    for i,val in enumerate(data):
        avg_nmad += val["nmad"]
        avg_bias += val["bias"]
        avg_MAE += val["MAE"]
        avg_frac += val["frac"]
        avg_loss += val['loss']
        z1.extend(val["zspec"])
        z2.extend(val["zphoto"])

    avg_nmad /= len(data)
    avg_bias /= len(data)
    avg_MAE /= len(data)
    avg_frac /= len(data)
    avg_loss /= len(data)
    plot_density(z1, z2, filename)
    return [avg_loss,avg_MAE,avg_nmad,avg_bias,avg_frac]

#------------------------------------------------------------------------------

#-------------------------------------------------------
class MyDataset(Dataset):
    def __init__(self, data, targets, mm=False, transform=None):
        self.data = data
        self.mm = mm # flag for mixed model input.
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index,:,:,:].copy()  # copy() is to avoid mmap error
        y = self.targets[index][5].copy()  # i.e. too many files open (>1024)
        if self.transform:
            x = self.transform(x)
        if self.mm == True:
            # append 5 extra information.
            x_0 = torch.Tensor([self.targets[index][i] for i in range(14,19)])
            return x,x_0,y
        return x,y

    def __len__(self):
        return self.data.shape[0]
#---------------------------------------------------------
def weight_init(m):
    '''
    Usage:
        model.apply(weight_init)
    '''
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param)
            else:
                nn.init.normal_(param)
#----------------------------------------------------------

import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

def plot_density(x, y, savepth=""):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')
    x9 =[i/100 for i in range(100)]
    y9 = [i/100+0.05 for i in range(100)]
    y10 = [i/100-0.05 for i in range(100)]
    ax.plot(x9,y9,color='magenta', linewidth=1.5, linestyle='--',alpha=0.8)
    ax.plot(x9,y10,color='magenta', linewidth=1.5, linestyle='--',alpha=0.8)

    ax.set_title("Zphoto vs zspec")
    ax.set_xlabel("zphoto")
    ax.set_xlim([0,0.60])
    ax.set_ylim([0,0.60])
    ax.set_ylabel("zspec")
    plt.savefig(savepth)

if __name__ == '__main__':
    # plot density test.
    x = np.abs(np.random.normal(size=100000))
    y = abs(x + 0.05*np.random.normal(size=100000))
    plot_density(x, y)
    plt.show()


