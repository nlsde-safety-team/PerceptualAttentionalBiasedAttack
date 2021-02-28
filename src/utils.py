import numpy as np
import os
from torchvision import transforms
from PIL import Image
import torch
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F



def get_pic_info(root,file_path):
    labelset = []
    pathset = []
    centerset = []
    bboxset = []
    lineset = []
    f = open(file_path, 'r')
    for line in f:
        lineset.append(line)
        line = line.rstrip()
        words = line.split()
        path = words[0]
        pathset.append(os.path.join(root, path))
        labelset.append(int(words[1]) - 1)
        centerset.append([float(words[2]),float(words[3])])
        bboxset.append([float(words[4]),float(words[5]),float(words[6]),float(words[7])])
    f.close()
    labelset = np.array(labelset)
    centerset = np.array(centerset)
    bboxset = np.array(bboxset)
    
    return labelset, pathset, centerset, bboxset, lineset

def gram_matrix(input):
    a, b, c, d = input.shape
    features = input.view(a, b, c * d)
    assert features[0][0][1] == features.transpose(1, 2)[0][1][0]

    
    G = torch.matmul(features, features.transpose(1, 2)) / b
    return G

def tensor_show(tensor, path):
    tensor = tensor * 0.5 + 0.5
    image = transforms.ToPILImage()(tensor)
    image.save(path)
    
def pad_transform(patch, image_size, patch_size, offset):
    # print(offset)
    offset_x, offset_y = offset

    pad = torch.nn.ConstantPad2d((offset_x - patch_size // 2, image_size- patch_size - offset_x + patch_size // 2, offset_y - patch_size // 2, image_size-patch_size-offset_y + patch_size // 2), 0) #left, right, top ,bottom
    mask = torch.ones((3, patch_size, patch_size)).cuda()
    return pad(patch), pad(mask)

def _gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = _create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = _create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
