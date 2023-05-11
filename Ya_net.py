import math
import datetime
import cv2
from torchvision.models import vgg16, VGG16_Weights
from torch import nn

import yaconf
from ZCLAlgorithm import yaclass
import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
from torch.autograd import Variable
import torch.fft as fft
from torchviz import make_dot



def gaussian(original_image, down_times):
    temp = original_image.copy()
    gaussian_pyramid = [temp]
    for i in range(down_times):
        temp = cv2.pyrDown(temp)
        gaussian_pyramid.append(temp)
    return gaussian_pyramid


def laplacian(gaussian_pyramid, up_times):
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(up_times, 0, -1):
        temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i])
        # temp_lap = cv2.subtract(gaussian_pyramid[i-1], temp_pyrUp)
        temp_lap = gaussian_pyramid[i - 1] - temp_pyrUp
        laplacian_pyramid.append(temp_lap)
    return laplacian_pyramid


def get_gram_matrix(f_map):
    n, c, h, w = f_map.shape
    if n == 1:
        f_map = f_map.reshape(c, h * w)
        gram_matrix = torch.mm(f_map, f_map.t())
        return gram_matrix
    else:
        f_map = f_map.reshape(n, c, h * w)
        gram_matrix = torch.matmul(f_map, f_map.transpose(1, 2))
        return gram_matrix


def load_image(path):
    image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image
class yadate(Dataset):
    def __init__(self, dataPathin, dataPathout):
        super(yadate, self).__init__()
        np.random.seed(0)
        self.datas, self.yalen = self.loadToMem(dataPathin, dataPathout)

    def loadToMem(self, dataPathin, dataPathout):
        datas = {}
        datas[0] = []
        datas[1] = []
        yalen = len(dataPathin)
        for idx in range(yalen):
            datas[0].append(load_image(dataPathin[idx]))
            datas[1].append(load_image(dataPathout[idx]))
        return datas, yalen

    def __len__(self):
        return self.yalen

    def __getitem__(self, index):
        image1 = self.datas[0][index].cuda()
        image2 = self.datas[1][index].cuda()
        return image1, image2

class yaya(nn.Module):
    def __init__(self):
        super(yaya, self).__init__()
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    def forward(self, input_):
        return self.vgg(input_)
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        a = vgg.features
        # print(a)
        self.layer1 = a[:10]  # 0-9
        self.rest1 = a[10:15]  # 10-14
        self.restichange1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1,
                                      bias=False)

        self.layer2 = a[15:17]  # 15-16
        self.rest2 = a[17:22]  # 17-21
        self.restichange2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1,
                                      bias=False)

        self.layer3 = a[22:24]  # 22-23
        self.rest3 = a[24:29]  # 24-28

        self.layer4 = a[29:]  # 29-30
        self.ya_convs = nn.Sequential(nn.Linear(in_features=512 * 16 * 16, out_features=256, bias=True),
                                      nn.Linear(in_features=256, out_features=128, bias=True),
                                      nn.Linear(in_features=128, out_features=3, bias=True),
                                      nn.Sigmoid())

    def forward(self, input_):
        out1 = self.layer1(input_)
        s1   = self.rest1(out1)
        out2 = self.layer2( s1 + self.restichange1(out1))
        s2   = self.rest2(out2)
        out3 = self.layer3( s2 + self.restichange2(out2))
        s3   = self.rest3(out3)
        out4 = self.layer4( s3 + out3)
        out5 = out4.view(-1, 512 * 16 * 16)
        out6 = self.ya_convs(out5)
       # return torch.sigmoid(s1),torch.sigmoid(s2),torch.sigmoid(s3),out6
        return out6
class yanet(nn.Module):
    def __init__(self):
        super(yanet, self).__init__()
        self.vgg = VGG16()
        self.loss_AVER = nn.L1Loss(reduction='mean').cuda()
        self.loss_XS   = nn.MSELoss(reduction='sum').cuda()
        self.loss_CF   = nn.L1Loss(reduction='mean').cuda()
        self.loss_STD  = nn.L1Loss(reduction='mean').cuda()
        #self.loss_S    = nn.L1Loss(reduction='mean').cuda()
        #self.loss_FFT  = nn.L1Loss(reduction='mean').cuda()
        self.ya_loss = 0
        self.batch_size = 8
        self.yaepoch = 5
        self.modelpath = 'modelpath\\Modelts'
        self.modelpath = 'modelpath\\Ya_Model'
        self.wloss_aver = 10  #均值
        self.wloss_xs   = 350    #像素均方误差
        self.wloss_cf   = 1    #参数变化率
        self.wloss_std  = 80    #标准差
        #self.wloss_s = 3       #格拉姆矩阵
        self.wloss_fft = 50   #fft
        self.train_tag = False
        self.train_show_tag = False
        self.save_tag = True
        self.load_tag = True
        self.out_nor = None
        if (self.load_tag):
            if (os.path.exists(self.modelpath)):
                self.yaload()
        self.h, self.w = 512, 512
        lpf = torch.zeros((self.h, self.w))
        R = 30
        for x in range(self.w):
            for y in range(self.h):
                if ((x - (self.w - 1) / 2) ** 2 + (y - (self.h - 1) / 2) ** 2) < (R ** 2):
                    lpf[y, x] = 1
        hpf = 1 - lpf
        self.hpf, self.lpf = hpf.cuda(), lpf.cuda()
    def forward(self, input):
        out  = self.vgg(input)#.type(torch.float)

        #outa = out[:, 0].unsqueeze(1) * 5 + 0.5
        #outb = out[:, 1].unsqueeze(1) + 0.99
        #outc = out[:, 2].unsqueeze(1) * 0.2 - 0.1
        outa = yaconf.yap(out[:, 0].unsqueeze(1), 1)
        outb = yaconf.yap(out[:, 1].unsqueeze(1), 2)
        outc = yaconf.yap(out[:, 2].unsqueeze(1), 3)
        # print(outa,outb,outc)
        Tersor_t = torch.ones([outa.shape[0]]).cuda().t()
        lut = torch.clamp(255*torch.pow(torch.linspace(0, 1, 256).cuda(), outa), 0, 255).t()
        out2 = None
        for i in range(input.shape[0]):
            if (i == 0):
                outtemp = (lut[:, i].unsqueeze(1)[(input[i, :, :, :].unsqueeze(0) * 255).type(torch.long)]).squeeze(4)
                at = outb[i]
                bt = 255 * outc[i]
                out_contrast = torch.clamp((at * outtemp + bt), 0, 255)
                out2 = out_contrast
            else:
                outtemp = (lut[:, i].unsqueeze(1)[(input[i, :, :, :].unsqueeze(0) * 255).type(torch.long)]).squeeze(4)
                at = outb[i]
                bt = 255 * outc[i]
                out_contrast = torch.clamp((at * outtemp + bt), 0, 255)
                out2 = torch.cat((out2, out_contrast), 0)
        out3 = out2 / 255.0
        if (self.train_tag):
            if (self.train_show_tag):
                self.showbatchs("ya", out3)
            loss_aver = self.loss_AVER(torch.mean(out3), torch.mean(self.out_nor)) * self.wloss_aver # 0-1

            loss_xs = self.loss_XS(out3, self.out_nor) /(input.shape[0]*786432) * self.wloss_xs # 0-1

            loss_cf = ((self.loss_CF(outa.squeeze(1), Tersor_t)-4.5)** 2)/20.25 * self.wloss_cf #0-1

            loss_std = self.loss_STD(torch.std(out3),torch.std(self.out_nor)) * self.wloss_std #0-1

            f1 = fft.fftn(out3, dim=(2, 3))
            f1 = torch.roll(f1, (self.h // 2, self.w // 2), dims=(2, 3))  # 移频操作,把低频放到中央
            f_h1 = f1 * self.hpf
            f_h1 = torch.abs(fft.ifftn(f_h1, dim=(2, 3)))
            X1 = (f_h1 * 255).type(torch.uint8)
            Y1 = torch.nonzero(X1 > 100, as_tuple=False).shape[0]

            f2 = fft.fftn(self.out_nor, dim=(2, 3))
            f2 = torch.roll(f2, (self.h // 2, self.w // 2), dims=(2, 3))  # 移频操作,把低频放到中央
            f_h2 = f2 * self.hpf
            f_h2 = torch.abs(fft.ifftn(f_h2, dim=(2, 3)))
            X2= (f_h2 * 255).type(torch.uint8)
            Y2 = torch.nonzero(X2 > 100, as_tuple=False).shape[0]
            Y= Y2 - Y1
            if(Y<0):
                loss_fft = 0
            else:
                loss_fft = Y/(input.shape[0]*786432)*self.wloss_fft
            #print()
            #print(loss_fft)
            self.ya_loss = loss_aver + loss_xs + loss_cf+loss_std +loss_fft
            return out3
        else:
            return out3
    def showbatchs(self, k, input):
        wsize  =300
        lensize=4
        tagfirst = True
        image = np.zeros([wsize, wsize, 3]).astype(np.uint8)
        count = 0
        countname = 0
        for i in range(input.shape[0]):
            temp_img = input[i, :, :, :]
            temp_img = temp_img.cpu().detach().numpy().transpose((1, 2, 0)) * 255
            temp_img = temp_img.astype(np.uint8)
            temp_img = cv2.resize(temp_img, (wsize, wsize))
            if (tagfirst):
                tagfirst = False
                image = temp_img
            else:
                image = np.concatenate([image, temp_img], axis=1)
            count += 1
            if (count == lensize):
                count = 0
                countname += 1
                if (i != input.shape[0] - 1):
                    cv2.imshow(k + str(countname), image)
                    image = np.zeros([wsize, wsize, 3]).astype(np.uint8)
                    tagfirst = True
        countname += 1
        cv2.imshow(k + str(countname), image)
        cv2.waitKey(1)

    def show(self, k, input):
        res_img = input.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0)) * 255
        res_img = res_img.astype(np.uint8)
        # res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
        cv2.imshow(k, res_img)

    def yashow(self, input1, input2, input3):
        self.show("input", input1)
        self.show("output", input2)
        self.show("norm", input3)
        cv2.waitKey(1)

    def yatrain(self, netin, netout, batchsize=3, epoch=5,batchcount=5,showtag=False):
        self.yaepoch=epoch
        self.train_tag = True
        self.train_show_tag = showtag
        self.batch_size = batchsize
        save_tag = False
        torch.backends.cudnn.benchmark = True
        #optimizer = torch.optim.Adam(self.vgg.parameters(), lr=0.00002)
        trainSet = yadate(netin, netout)
        datalen = len(trainSet)
        if (datalen >= 400 or epoch>=5):
            save_tag = True
        overtag = datalen // self.batch_size
        trainLoader = DataLoader(trainSet, batch_size=self.batch_size, shuffle=False)
        losssum = self.wloss_aver + self.wloss_xs + self.wloss_cf + self.wloss_std + self.wloss_fft
        losslist = []
        yalr = 0.00002
        loss_aft = 1
        yalr_mu = 0.1
        handingtime = 0
        start_time = datetime.datetime.now()

        for i in range(self.yaepoch):
            loss=0
            losscount=0
            for batch_id, (img1s, img2s) in enumerate(trainLoader, 1):
                optimizer = torch.optim.Adam(self.vgg.parameters(), lr=0.00002)
                count_t = int(batch_id / overtag * 10)
                strshow = '#' * count_t
                img1s = img1s.reshape([-1, 3, 512, 512])
                img2s = img2s.reshape([-1, 3, 512, 512])
                self.out_nor = img2s
                for j in range(batchcount):
                    self.forward(img1s)
                    optimizer.zero_grad()
                    self.ya_loss.backward()
                    optimizer.step()
                    losstemp=self.ya_loss.cpu().detach().numpy()/losssum
                    
                    #end_time = datetime.datetime.now()
                    #print((end_time - start_time).seconds)

                    losslist.append(losstemp)
                    loss+=losstemp
                    losscount+=1
                    print("\rEpoch{}--[{}],loss=[{}],lr=[{}]" \
                          .format(i, strshow.ljust(10, '+'), losstemp,yalr), end="")
            loss=loss/losscount
            if (abs(loss_aft - loss) <= yalr_mu):
                yalr_mu /= 2
                yalr /= 2
            loss_aft = loss
            print("\nEpoch{}-LOSS:{}\n" \
                    .format(i, loss, end=""))

            if (save_tag):
                self.yasave(i)
                f = open("savepath\\losslist{}.txt".format(i), "w")
                for line in losslist:
                    f.write(str(line) + '\n')
                f.write(str(loss) + '\n')
                f.close()
                losslist = []

            end_time = datetime.datetime.now()

            if ((end_time - start_time).seconds >= 60):
                pass
                #os.system("shutdown -s -t 0")
        self.yasave("_over_range_"+str(self.yaepoch))
        return
        os.system("shutdown -s -t 0")
    def yapredict(self, path, batchsize=1):
        self.train_tag = False
        self.batch_size = batchsize
        img = load_image(path)
        img = Variable(img.cuda())
        self.show("in", img)
        self.show("out", self.forward(img))
        cv2.waitKey(0)

    def predict(self, input, batchsize=1):
        self.train_tag = False
        self.batch_size = batchsize
        image = cv2.resize(input, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = torch.from_numpy(image).float() / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        img = Variable(image.cuda())
        res_img = self.forward(img).cpu().detach().numpy().squeeze(0).transpose((1, 2, 0)) * 255
        res_img = res_img.astype(np.uint8)
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        return res_img
    def apppredict(self, input, batchsize=1):
        self.train_tag = False
        self.batch_size = batchsize
        image = cv2.resize(input, (512, 512))
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = torch.from_numpy(image).float() / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        img = Variable(image.cuda())
        res_img = self.forward(img).cpu().detach().numpy().squeeze(0).transpose((1, 2, 0)) * 255
        res_img = res_img.astype(np.uint8)
        #res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        return res_img
    def yaload(self):
        state_dict = torch.load(self.modelpath)
        self.vgg.load_state_dict(state_dict['model'])

    def yasave(self,epoch=''):
        if (self.save_tag):
            savepath="{}_Epoch{}".format(self.modelpath,str(epoch))
            torch.save({'model': self.vgg.state_dict()}, savepath)
def makedata(fi,f1,f2,f3,f4,e1,e2,k=False):
    input  = []
    output = []
    for i in range(len(e1)):
        input.append(e1[i])
        output.append(e2[i])
    for i in range(len(fi)):
        output.append(fi[i])
        t = random.randint(1, 4)
        if (t==1):
            input.append(f1[i])
        elif (t == 2):
            input.append(f2[i])
        elif (t == 3):
            input.append(f3[i])
        elif (t == 4):
            input.append(f4[i])
    if(k==True):
        for i in range(len(fi)//2):
            input.append(fi[i])
            output.append(fi[i])
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(input)
    random.seed(randnum)
    random.shuffle(output)
    return input,output
def datashow(input,output):
    for i in range(len(input)):
        c1=cv2.imread(input [i])
        c2=cv2.imread(output[i])
        c2=cv2.resize(c2,(c1.shape[1],c1.shape[0]))
        imag = np.concatenate([c1,c2], axis=1)
        cv2.imshow("k",imag)
        cv2.waitKey(80)
if __name__ == "__main__":
    net = yanet().cuda()
    #yavgg=yaya().cuda()
    #data = torch.randn(1, 3, 512, 512).cuda()
    #g = make_dot(yavgg(data))
    #g.view()
    #while 1:
    #    pass
    count = 200
    count = count//2
    c = yaclass()
    c.yaloadkinds()
    c.yaloadlol()
    input, output = makedata(c.fi[:count], c.fla[:count], c.flb[:count], c.fha[:count], c.fhb[:count], c.lolfl[:count], c.lolfh[:count],True)
    #input, output  =makedata(c.fi,c.fla,c.flb,c.fha,c.fhb,c.lolfl,c.lolfh,True)
    #datashow(input,output)
    net.yatrain(input,output, batchsize=8, epoch=50,batchcount=2,showtag=True)
    #net.yapredict('D:\\zclbs\\daima\\data\\2.bmp')
