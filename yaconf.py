import cv2
import torch
import numpy as np

model5count=100
inputsize=512
fstr=r'data/2.bmp'
fui=r"GUI/bs.ui"
inputfs=r'data_input'
outputfs=r'data_output'
cam=(0,'192.168.31.138','4747')

model2base=r"D:\zclbs\daima\mode\moderetinex_net"
Retinex_netpath=model2base+r"\ckpts/"

model6path=r'D:\zclbs\daima\mode\modezero_reference\snapshots/Epoch99.pth'
device = 'cuda'
lr = 0.001
iterations = 100
illu_factor = 1
reflect_factor = 1
noise_factor = 5000
reffac = 1
gamma = 0.4
g_kernel_size = 5
g_padding = 2
sigma = 3
kx = cv2.getGaussianKernel(g_kernel_size,sigma)
ky = cv2.getGaussianKernel(g_kernel_size,sigma)
gaussian_kernel = np.multiply(kx,np.transpose(ky))
gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0).to(device)

model7base=r"D:\zclbs\daima\mode\modeuretinex_net"
yDecom_model_low_path=model7base+"/ckpt/init_low.pth"
yunfolding_model_path=model7base+"/ckpt/unfolding.pth"
yadjust_model_path=model7base+"/ckpt/L_adjust.pth"
yratio=5
class yat():
    def __init__(self,Decom_model_low_path,unfolding_model_path,adjust_model_path,ratio):
        self.Decom_model_low_path=Decom_model_low_path
        self.unfolding_model_path=unfolding_model_path
        self.adjust_model_path=adjust_model_path
        self.ratio=ratio

def yap(input,k):
    if (k == 1):
        return input * 5 + 0.5
    elif (k == 2):
        return input + 1
    elif (k == 3):
        return input * 0.2 - 0.1
    elif (k == 4):
        return input
    elif (k == 5):
        return (input / 2)/10
