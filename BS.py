'''
File Name          : BS.py
Author             : CUTEYAYA
Version            : V1.0.15
Created on         : 2023/2/27
'''
#1搭建基础框架
#2优化线程堵塞闪退
#3增加评价指标1、2
#4增加评价指标3、4
#5增加评价指标5、6
#6增加线程判断运行状态
#7增加模型3 Retinex_net
#8增加自适应直方图均衡化
#9增加数据组识别,摄像头识别
#10增加命令行带参数
#11优化数据组输出 优化摄像头输入
#12增加模型5 rrdnet 优化参数输入
#13增加模型6 zero_reference 增加模型7 uretinexnet
#14增加自己的算法
#15增加模型选择
#16增加视频识别
import glob
import math
import sys
import threading
import cv2
from PySide6 import QtGui
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage import exposure
import numpy as np
import os
from mode.moderetinex_net.model import RetinexNet
import argparse
import warnings
import yaconf as conf
from mode.moderrd_net.RRDNet import RRDNet
from mode.moderrd_net.pipline import pipline_retinex
import torch
import mode.modezero_reference.model as model6
from mode.modeuretinex_net.test import Inference
from mode.modezero_reference.model6predit import lowlight
from Ya_net import yanet

#import ZCLAlgorithm
print("openGUI")
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser(description='')
parser.add_argument('--fstr', dest='fstr', default=conf.fstr,
                    help='单张输入路径')
parser.add_argument('--inputsize', dest='inputsize', type=int, default=conf.inputsize,
                    help='输入图片格式化大小')
parser.add_argument('--fui', dest='fui', default=conf.fui,
                    help='GUI载入路径')
parser.add_argument('--inputfs', dest='inputfs',default=conf.inputfs,
                    help='图片组输入路径')
parser.add_argument('--outputfs', dest='outputfs',default=conf.outputfs,
                    help='结果保存路径')
parser.add_argument('--cam', dest='cam',  nargs='+',default=conf.cam,  help='(0 X) 电脑摄像头 or (1 http port) 网络摄像头')
parser.add_argument('--Retinex_netpath', dest='Retinex_netpath',default=conf.Retinex_netpath,
                    help='Retinex-net下载路径')
args = parser.parse_args()
fstr=args.fstr
fui=args.fui
fstr1=args.inputfs
fstr2=args.outputfs
yasize=args.inputsize
Retinex_netpath=args.Retinex_netpath
model5count=conf.model5count

Yanet =yanet().cuda()
def yamode(input,k):
    if (k == 1):
        out = Yanet.predict(input)
    elif (k == 2):
        out = Yanet.predict(cv2.merge([input, input, input]))
    else:
        out = Yanet.predict(input)
        out = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
    return out

if(tuple(args.cam)[0]==0):
    yacam=0
else:
    yacam = 'http://{0}:{1}/video'.format(tuple(args.cam)[1], tuple(args.cam)[2])
def yapj1(*args):#均值 标准差 越大表明图像明暗渐变层越多，图像细节越突出越清晰
    input = args[0]
    input = input.astype(np.float32)
    mean_img = input
    l1=mean_img.mean()
    mean_img -= mean_img.mean()
    std_img = mean_img.copy()
    l2 =std_img.std()
    return  '{:.2f}'.format(l1),'{:.2f}'.format(l2)
def yapj2(*args):#熵越大，图像越清晰，信息越丰富
    input = args[0]
    l1=input.shape[0]
    l2=input.shape[1]
    res = 0
    tmp= [0 for i in range (0,256)]
    for i in range(l1):
        for j in range(l2):
            val = input[i][j]
            tmp[val] = tmp[val] + 1
    k=l1*l2
    for i in range(256):
        tmp[i] = float(tmp[i] / k)
    for i in range(256):
        if(tmp[i] != 0):
            res = res - tmp[i] * (math.log(tmp[i]) / math.log(2))
    return '{:.2f}'.format(res)
def yapj3(*args):#PSNR越大，代表着图像质量越好
    input = args[0]
    output = args[1]
    psnr = compare_psnr(input, output)
    return '{:.2f}'.format(psnr)
def yapj4(*args):#SSIM取值范围为[0,1]，值越大表示输出图像和无失真图像的差距越小，即图像质量越好
    input = args[0]
    output = args[1]
    ssim = compare_ssim(input, output)
    return '{:.2f}'.format(ssim)
def yapj5(*args):#图像的清晰度和纹理变化，平均梯度越大说明图像越清晰
    input = args[0]
    input = input.astype(np.float16)/255.0
    tmp = 0.0
    rows = input.shape[0] - 1
    cols = input.shape[1]  - 1
    for i in range(rows):
        for j in range(cols):
            t=input[i,j]
            dx = input[i, j + 1] - t
            dy = input[i + 1,j] - t
            ds = np.sqrt((dx*dx + dy*dy) / 2)#######超出数据范围
            tmp += ds
    imageAvG = tmp / (rows*cols)
    return '{:.2f}'.format(imageAvG)
def mode1(input,k):
    def mode1ts(I):
        F = cv2.GaussianBlur(I, (3, 3), 0)  # 高斯模糊
        I[I == 0] = 1
        F[F == 0] = 1
        LogI = cv2.log(I / 255.0)
        LogF = cv2.log(F / 255.0)
        LogL = cv2.multiply(LogI, LogF)
        logR = cv2.subtract(LogI, LogL)
        R = cv2.normalize(logR, None, 0, 255, cv2.NORM_MINMAX)
        log_uint8 = cv2.convertScaleAbs(R)
        return log_uint8
    hs=sys._getframe().f_code.co_name+'ts'
    if(k==3):
        b_gray, g_gray, r_gray = cv2.split(input)
        r_gray=eval(hs)(r_gray)
        out = cv2.merge([b_gray, g_gray, r_gray]) #色调(H)、饱和度(S).亮度(V).
        out = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
    elif(k==1):
        b_gray, g_gray, r_gray = cv2.split(input)
        out1 = eval(hs)(b_gray)
        out2 = eval(hs)(g_gray)
        out3 = eval(hs)(r_gray)
        out = cv2.merge([out1, out2, out3])
    else:
        out = eval(hs)(input)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    return out
model2 = RetinexNet().cuda()
def mode2(input,k):
    hs=sys._getframe().f_code.co_name+'ts'
    if (k == 1):
        out=model2.predict(input,ckpt_dir=Retinex_netpath)
    elif (k == 2):
        out = cv2.merge([input, input, input])
        out = model2.predict(out,ckpt_dir=Retinex_netpath)
        #out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    else:
        out = model2.predict(input,ckpt_dir=Retinex_netpath)
        out = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
    return out
def mode3(input,k):
    def mode3ts(img, z_max=255):
        H, W = img.shape
        S = H * W * 1.
        out = img.copy()
        sum_h = 0.
        for i in range(1, 255):
            ind = np.where(img == i)
            sum_h += len(img[ind])
            z_prime = z_max / S * sum_h
            if (z_prime > 255):
                z_prime = 255
            elif (z_prime < 0):
                z_prime = 0
            out[ind] = z_prime * 1.0
        out = out.astype(np.uint8)
        return out
    hs=sys._getframe().f_code.co_name+'ts'
    if (k == 3):
        b_gray, g_gray, r_gray = cv2.split(input)
        r_gray = eval(hs)(r_gray)
        out = cv2.merge([b_gray, g_gray, r_gray])
        out = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
    elif (k == 1):
        b_gray, g_gray, r_gray = cv2.split(input)
        out1 = eval(hs)(b_gray)
        out2 = eval(hs)(g_gray)
        out3 = eval(hs)(r_gray)
        out = cv2.merge([out1, out2, out3])
    else:
        out = eval(hs)(input)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    return out
def mode4(input,k):
    def mode4ts(img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    hs=sys._getframe().f_code.co_name+'ts'
    if (k == 1):
        b_gray, g_gray, r_gray = cv2.split(input)
        out1 = eval(hs)(b_gray)
        out2 = eval(hs)(g_gray)
        out3 = eval(hs)(r_gray)
        out = cv2.merge([out1, out2, out3])
    elif (k == 2):
        out = eval(hs)(input)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    else:
        ycrcb = cv2.cvtColor(input, cv2.COLOR_HSV2BGR)
        ycrcb = cv2.cvtColor(ycrcb, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe.apply(channels[0], channels[0])
        cv2.merge(channels, ycrcb)
        out=cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)
    return out
model5 = RRDNet()
model5 = model5.to(conf.device)
def mode5(input,k):
    hs=sys._getframe().f_code.co_name+'ts'
    if (k == 1):
        res_img= pipline_retinex(model5, input,model5count)
        out=res_img
    elif (k == 2):
        out = cv2.merge([input, input, input])
        res_img = pipline_retinex(model5, out, model5count)
        out = res_img
    else:
        res_img = pipline_retinex(model5, input, model5count)
        out = cv2.cvtColor(res_img, cv2.COLOR_HSV2RGB)
    return out
model6 = model6.enhance_net_nopool().cuda()
model6.load_state_dict(torch.load(conf.model6path))
def mode6(input,k):
    hs=sys._getframe().f_code.co_name+'ts'
    if (k == 1):
        out = lowlight(input,model6)
        #print(out.shape)
    elif (k == 2):
        out = cv2.merge([input, input, input])
        out = lowlight(out,model6)
    else:
        out = lowlight(input, model6)
        out = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
    return out
yat=conf.yat(conf.yDecom_model_low_path,conf.yunfolding_model_path,conf.yadjust_model_path,conf.yratio)
model7 = Inference(yat).cuda()
def mode7(input,k):
    hs=sys._getframe().f_code.co_name+'ts'
    if (k == 1):
        out=model7.run(input)
    elif (k == 2):
        out = cv2.merge([input, input, input])
        out=model7.run(out)
    else:
        out = model7.run(input)
        out = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
    return out
def yaway1(*args):
    input=args[0]
    cs=args[1]
    k=args[2]
    if(k==1 or k==3):
        rows, cols, channels = input.shape
        blank = np.ones([rows, cols, channels], input.dtype)
        if(cs>=1):
            blank=blank*255
            rst = cv2.addWeighted(input, 2 - cs, blank,cs-1 , 0)
        else:
            rst = cv2.addWeighted(input, cs, blank, 1 - cs, 0)
    else:
        rows, cols = input.shape
        blank = np.ones([rows, cols], input.dtype)
        rst = cv2.addWeighted(input, cs, blank, 1 - cs, 0)
    return rst
def yaway2(*args):
    img = args[0]
    cs = args[1]
    k = args[2]
    cs=abs(cs-2)
    out = exposure.adjust_gamma(img, cs)  # 调暗
    return out
def yaway3(*args):                        #直方图均衡化增加噪声
    img=args[0]
    cs=args[1]
    k = args[2]
    def eq(input,cs):
        H, W = input.shape
        S = H * W * 1.
        out = input.copy()
        sum_h = 0.
        cs=(cs+0.001)*10/3
        for i in range(1, 256):
            ind = np.where(input == i)
            sum_h += len(input[ind])
            z_prime = 255 / S * sum_h*(-math.log((256-i)/255)*cs)
            if (z_prime > 255):
                z_prime = 255
            elif (z_prime < 0):
                z_prime = 0
            out[ind] = z_prime * 1.0

        out = out.astype(np.uint8)
        return out
    if (k == 1 or k == 3):
        b_gray, g_gray, r_gray = cv2.split(img)
        b_gray = eq(b_gray, cs)
        g_gray = eq(g_gray, cs)
        r_gray = eq(r_gray,cs)
        out = cv2.merge([b_gray, g_gray, r_gray])
    else:
        out=eq(img,cs)
    return out
class yaxc(QObject):
    ya1 = Signal(QPixmap,int)
    ya2 = Signal(QPixmap,int)
class Stats:
    def __init__(self):
        self.hsaffect = ''
        self.hsmode='mode1'

        self.I =''
        self.I1=''
        self.I2=''
        self.I3=''

        self.sbpd=0

        self.f1=1
        self.f2=1
        self.f3=0.5
        self.fa = 0
        self.fb = 0
        self.fc = 0

        self.fstr = fstr
        self.fstr1 = fstr1
        self.fstr2 = fstr2

        self.ui = QUiLoader().load(fui)
        self.ui.model5count.setValidator(QtGui.QIntValidator())
        self.ui.model5count.setText(str(model5count))
        self.ui.af1.setChecked(True)
        self.ui.fa.setRange(0, 2000)
        self.ui.fa.setValue(1000)
        self.ui.fb.setRange(0, 2000)
        self.ui.fb.setValue(1000)
        self.ui.fc.setRange(0, 1000)
        self.ui.fc.setValue(500)
        self.ui.f1.setText(str(self.f1))
        self.ui.f2.setText(str(self.f2))
        self.ui.f3.setText(str(self.f3))
        self.afdisenable()

        self.ui.FBti.clicked.connect(self.choose)
        self.ui.FBtsi.clicked.connect(self.choose1)
        self.ui.FBtso.clicked.connect(self.choose2)
        self.ui.Fnetmodel.clicked.connect(self.choose_net)

        self.ui.bt.clicked.connect(self.handling)
        self.ui.bts.clicked.connect(self.handlings)
        self.ui.btcam.clicked.connect(self.handlingcam)

        self.ui.fa.sliderReleased.connect(self.yaf1)
        self.ui.fb.sliderReleased.connect(self.yaf2)
        self.ui.fc.sliderReleased.connect(self.yaf3)
        self.ck= QFileDialog(self.ui)
        self.ui.img.setScaledContents(True)
        self.ui.img.setFixedSize(500, 400)
        self.ui.result.setScaledContents(True)
        self.ui.result.setFixedSize(500, 400)
        self.ui.lineEdit.setText(self.fstr)

        self.ui.mode1.clicked.connect(self.pdmode)
        self.ui.mode2.clicked.connect(self.pdmode)
        self.ui.mode3.clicked.connect(self.pdmode)
        self.ui.mode4.clicked.connect(self.pdmode)
        self.ui.mode5.clicked.connect(self.pdmode)
        self.ui.mode6.clicked.connect(self.pdmode)
        self.ui.mode7.clicked.connect(self.pdmode)
        self.ui.yamode.clicked.connect(self.pdmode)

        self.ui.affect.toggled.connect(self.afzdhs)
        self.ui.af1.clicked.connect(self.af123)
        self.ui.af2.clicked.connect(self.af123)
        self.ui.af3.clicked.connect(self.af123)

        self.ui.rgb.clicked.connect(self.zdhs)
        self.ui.gray.clicked.connect(self.zdhs)
        self.ui.hsv.clicked.connect(self.zdhs)
        self.ui.zd.clicked.connect(self.zdhandling)
        self.ui.pj.clicked.connect(self.handling)

        self.ui.lineEdit1.setText(self.fstr1)
        self.ui.lineEdit2.setText(self.fstr2)
        self.ui.model5count.editingFinished.connect(self.model5change)
        self.ui.model5count.textChanged.connect(self.model5change)
        self.ui.model5count.selectionChanged.connect(self.model5change)

        self.ui.btcamclose.clicked.connect(self.handlingcamclose)
        self.camclosetag=True

        self.signals = yaxc()
        self.signals.ya1.connect(self.pF1)
        self.signals.ya2.connect(self.pF2)
        self.over()
        self.hthr  = threading.Thread(target=self.handlingthread)
        self.hthr.setDaemon(True)
        self.MessageBox = QMessageBox()

        self.videotag=False
    def yaread(self, input):
        #try:
            if(input.split("/")[-1].split(".")[1]=='mp4'):
                self.videotag = True
            else:
                self.videotag = False
                self.I = cv2.imread(input)
                height, width = self.I.shape[0], self.I.shape[1]
                if(height>width):
                    self.I = cv2.rotate(self.I, cv2.ROTATE_90_CLOCKWISE)
                    height, width=  width,height
                bl=height*1.0/width
                yawidth=yasize
                self.I = cv2.resize( self.I, (yawidth, int(bl*yawidth)))
                self.I1 = cv2.cvtColor(self.I, cv2.COLOR_BGR2RGB)
                self.I2 = cv2.cvtColor(self.I, cv2.COLOR_BGR2GRAY)
                self.I3 = cv2.cvtColor(self.I, cv2.COLOR_BGR2HSV)

                input = self.I
                # print(input)
                try:
                    showImage = QImage(input.data, input.shape[1], input.shape[0], QImage.Format_BGR888)
                    self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
                    self.signals.ya2.emit(QPixmap.fromImage(showImage), 0)
                except:
                    print("erro_over")
                    print(input)
        #except:
        #    print("error")
    def pF1(self, F1,k):
        if (k == 0):
            self.ui.img.clear()
        else:
            self.ui.img.setPixmap(F1)
    def pF2(self, F2,k):
        if (k == 0):
            self.ui.result.clear()
        else:
            self.ui.result.setPixmap(F2)

    def zdhandling(self):
        if (self.ui.zd.isChecked()):
            self.handling()
    def model5change(self):
        self.ui.mode5.setChecked(True)
    def pjini(self):
        self.ui.pj1a.setText("")
        self.ui.pj1b.setText("")
        self.ui.pj2a.setText("")
        self.ui.pj2b.setText("")

        self.ui.pj3.setText("")
        self.ui.pj4.setText("")

        self.ui.pj5a.setText("")
        self.ui.pj5b.setText("")
        self.ui.pj6a.setText("")
        self.ui.pj6b.setText("")

        self.ui.pjqh1.setText("")
        self.ui.pjqh2.setText("")
        self.ui.pjqh3.setText("")
        self.ui.pjqh4.setText("")
        self.ui.pjqh5.setText("")
        self.ui.pjqh6.setText("")
    def zdhs(self):
        if (self.ui.zd.isChecked()):
            self.handling()
    def af123(self):
        if (self.ui.af1.isChecked()):
            self.ui.f1.setEnabled(True)
            self.ui.fa.setEnabled(True)
            self.ui.f2.setEnabled(False)
            self.ui.fb.setEnabled(False)
            self.ui.f3.setEnabled(False)
            self.ui.fc.setEnabled(False)
        elif(self.ui.af2.isChecked()):
            self.ui.f1.setEnabled(False)
            self.ui.fa.setEnabled(False)
            self.ui.f2.setEnabled(True)
            self.ui.fb.setEnabled(True)
            self.ui.f3.setEnabled(False)
            self.ui.fc.setEnabled(False)
        else:
            self.ui.f1.setEnabled(False)
            self.ui.fa.setEnabled(False)
            self.ui.f2.setEnabled(False)
            self.ui.fb.setEnabled(False)
            self.ui.f3.setEnabled(True)
            self.ui.fc.setEnabled(True)
        if (self.ui.zd.isChecked()):
            self.handling()
    def afdisenable(self):
        self.ui.af1.setEnabled(False)
        self.ui.af2.setEnabled(False)
        self.ui.af3.setEnabled(False)
        self.ui.f1.setEnabled(False)
        self.ui.f2.setEnabled(False)
        self.ui.f3.setEnabled(False)
        self.ui.fa.setEnabled(False)
        self.ui.fb.setEnabled(False)
        self.ui.fc.setEnabled(False)
    def afzdhs(self):
        if (self.ui.affect.isChecked()):
            self.ui.af1.setEnabled(True)
            self.ui.af2.setEnabled(True)
            self.ui.af3.setEnabled(True)
            self.af123()
        else:
            self.afdisenable()

        if (self.ui.zd.isChecked()):
            self.handling()
    def pj(self,input,output):
        if (self.ui.pj.isChecked()):
            input = cv2.resize(input, (output.shape[1],output.shape[0]))
            inputgray = input
            outputgray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
            tag=0
            if(self.ui.pj1b.text()!=""):
                tag=1
                pjtemp1 = float(self.ui.pj1b.text())
                pjtemp2 = float(self.ui.pj2b.text())
                pjtemp3 = float(self.ui.pj3.text())
                pjtemp4 = float(self.ui.pj4.text())
                pjtemp5 = float(self.ui.pj5b.text())
                pjtemp6 = float(self.ui.pj6b.text())
            li = yapj1(inputgray)
            lo = yapj1(outputgray)
            self.ui.pj1a.setText(str(li[0]))
            self.ui.pj1b.setText(str(lo[0]))

            self.ui.pj2a.setText(str(yapj2(inputgray)))
            self.ui.pj2b.setText(str(yapj2(outputgray)))
            self.ui.pj3.setText(str(yapj3(inputgray,outputgray)))
            self.ui.pj4.setText(str(yapj4(inputgray, outputgray)))

            self.ui.pj5a.setText(str(li[1]))
            self.ui.pj5b.setText(str(lo[1]))

            self.ui.pj6a.setText(str(yapj5(inputgray)))
            self.ui.pj6b.setText(str(yapj5(outputgray)))

            if(tag):
                pjtemp1 = pjtemp1 - float(self.ui.pj1b.text())
                pjtemp2 = pjtemp2 - float(self.ui.pj2b.text())
                pjtemp3 = pjtemp3 - float(self.ui.pj3.text())
                pjtemp4 = pjtemp4 - float(self.ui.pj4.text())
                pjtemp5 = pjtemp5 - float(self.ui.pj5b.text())
                pjtemp6 = pjtemp6 - float(self.ui.pj6b.text())
                if(pjtemp1>0):
                    self.ui.pjqh1.setText("减少")
                elif(pjtemp1<0):
                    self.ui.pjqh1.setText("增加")
                else:
                    self.ui.pjqh1.setText("不变")

                if (pjtemp2 > 0):
                    self.ui.pjqh2.setText("减少")
                elif (pjtemp2 < 0):
                    self.ui.pjqh2.setText("增加")
                else:
                    self.ui.pjqh2.setText("不变")

                if (pjtemp3 > 0):
                    self.ui.pjqh3.setText("减少")
                elif (pjtemp3 < 0):
                    self.ui.pjqh3.setText("增加")
                else:
                    self.ui.pjqh3.setText("不变")

                if (pjtemp4 > 0):
                    self.ui.pjqh4.setText("减少")
                elif (pjtemp4 < 0):
                    self.ui.pjqh4.setText("增加")
                else:
                    self.ui.pjqh4.setText("不变")

                if (pjtemp5 > 0):
                    self.ui.pjqh5.setText("减少")
                elif (pjtemp5 < 0):
                    self.ui.pjqh5.setText("增加")
                else:
                    self.ui.pjqh5.setText("不变")

                if (pjtemp6 > 0):
                    self.ui.pjqh6.setText("减少")
                elif (pjtemp6 < 0):
                    self.ui.pjqh6.setText("增加")
                else:
                    self.ui.pjqh6.setText("不变")
        else:
            self.pjini()
    def pdmode(self):
        global model5count
        if(self.ui.mode1.isChecked()):
            self.hsmode='mode1'
        elif(self.ui.mode2.isChecked()):
            self.hsmode='mode2'
        elif (self.ui.mode3.isChecked()):
            self.hsmode = 'mode3'
        elif (self.ui.mode4.isChecked()):
            self.hsmode = 'mode4'
        elif (self.ui.mode5.isChecked()):
            self.hsmode = 'mode5'
            model5count = int(self.ui.model5count.text())
        elif (self.ui.mode6.isChecked()):
            self.hsmode = 'mode6'
        elif (self.ui.mode7.isChecked()):
            self.hsmode = 'mode7'
        elif (self.ui.yamode.isChecked()):
            self.hsmode = 'yamode'
        if(self.ui.zd.isChecked()):
            self.handling()
    def over(self):#显示原图
        self.yaread(self.ui.lineEdit.text())

    def choose_net(self):
        FileDirectory = self.ck.getOpenFileName(self.ui, "请选择要下载的Ya模型")
        if (FileDirectory[0] == ''):
            return
        self.ui.yanetpath.setText(FileDirectory[0])
        global Yanet
        Yanet.modelpath=FileDirectory[0]
        Yanet.yaload()
        self.over()
        if (self.ui.zd.isChecked()):
            self.handling()
    def choose(self):
        FileDirectory = self.ck.getOpenFileName(self.ui, "请选择要处理的单张图像or视频")
        if(FileDirectory[0]==''):
            return
        self.ui.lineEdit.setText(FileDirectory[0])
        self.over()
        if (self.ui.zd.isChecked()):
            self.handling()
    def choose1(self):
        FileDirectory = self.ck.getExistingDirectory(self.ui, "请选择输入数据集路径")
        if (FileDirectory == ''):
            return
        self.ui.lineEdit1.setText(FileDirectory)
    def choose2(self):
        FileDirectory = self.ck.getExistingDirectory(self.ui, "请选择存放路径")
        if (FileDirectory == ''):
            return
        self.ui.lineEdit2.setText(FileDirectory)

    def handlingcam(self):
        if (self.hthr.is_alive() == False):
            self.hthr = threading.Thread(target=self.handlingcamthread)
            self.hthr.setDaemon(True)
            self.hthr.start()
        else:
            self.showbusy()
    def handlingcamthread(self):
        cap = cv2.VideoCapture(yacam)
        self.camclosetag=False
        while cap.isOpened():
            ok, frame = cap.read()  # 读取一帧数据
            if not ok:
                break
            try:
                self.I = frame
                height, width = self.I.shape[0], self.I.shape[1]
                if (height > width):
                    self.I = cv2.rotate(self.I, cv2.ROTATE_90_CLOCKWISE)
                    height, width = width, height
                bl = height * 1.0 / width
                yawidth = yasize
                self.I = cv2.resize(self.I, (yawidth, int(bl * yawidth)))
                self.I1 = cv2.cvtColor(self.I, cv2.COLOR_BGR2RGB)
                self.I2 = cv2.cvtColor(self.I, cv2.COLOR_BGR2GRAY)
                self.I3 = cv2.cvtColor(self.I, cv2.COLOR_BGR2HSV)
            except:
                print("error")
            out = self.handlingthread()
            #output = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            #cv2.imshow("ya", output)
            #cv2.waitKey(1)
            if self.camclosetag:
                break
        cap.release()
        cv2.destroyAllWindows()
    def handlingcamclose(self):
        self.camclosetag=True

    def handlings(self):
        if (self.hthr.is_alive() == False):
            self.hthr = threading.Thread(target=self.handlingsthread)
            self.hthr.setDaemon(True)
            self.hthr.start()
        else:
            self.showbusy()
    def handlingsthread(self):
        input=self.ui.lineEdit1.text()
        f = glob.glob(input + "/*.jpg") + \
            glob.glob(input + "/*.bmp") + \
            glob.glob(input + "/*.png")
        for jpg in f:
            self.yaread(jpg)
            out=self.handlingthread()
            output = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            ftemp=self.ui.lineEdit2.text()+"\\"+self.hsmode
            if not os.path.exists(ftemp):
                os.makedirs(ftemp)
            cv2.imwrite(ftemp+"//out"+jpg.split('\\')[-1],output)
    def handling(self):
        if(self.hthr.is_alive()==False):
            self.hthr = threading.Thread(target=self.handlingthread)
            self.hthr.setDaemon(True)
            self.hthr.start()
        else:
            self.showbusy()
    def yavideo(self):
        cap = cv2.VideoCapture(self.ui.lineEdit.text())
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outvideo = cv2.VideoWriter('output.mp4', fourcc, 60.0, (512,512))
        self.camclosetag = False
        while cap.isOpened():
            ok, frame = cap.read()  # 读取一帧数据
            if not ok:
                break
            try:
                self.I = frame
                self.I = cv2.resize(self.I, (512, 512))
                self.I1 = cv2.cvtColor(self.I, cv2.COLOR_BGR2RGB)
                self.I2 = cv2.cvtColor(self.I, cv2.COLOR_BGR2GRAY)
                self.I3 = cv2.cvtColor(self.I, cv2.COLOR_BGR2HSV)
            except:
                print("error")
            #out = self.handlingthread()
            if (self.ui.affect.isChecked()):
                if (self.ui.af1.isChecked()):
                    cs = self.fa
                    self.hsaffect = "yaway1"
                elif (self.ui.af2.isChecked()):
                    cs = self.fb
                    self.hsaffect = "yaway2"
                elif (self.ui.af3.isChecked()):
                    cs = self.fc
                    self.hsaffect = "yaway3"
                else:
                    cs = 0
                    pass
                if (self.ui.rgb.isChecked()):
                    input = self.I1
                    input = eval(self.hsaffect)(input, cs, 1)
                    out = eval(self.hsmode)(input, 1)
                    inputpj = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
                    showImage = input
                    showImage = QImage(showImage.data, showImage.shape[1], showImage.shape[0], QImage.Format_RGB888)
                    self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
                elif (self.ui.gray.isChecked()):
                    gray = self.I2
                    input = eval(self.hsaffect)(gray, cs, 2)
                    out = eval(self.hsmode)(input, 2)
                    inputpj = input
                    showImage = input
                    showImage = QImage(showImage.data, showImage.shape[1], showImage.shape[0], QImage.Format_Grayscale8)
                    self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
                else:
                    input = self.I3
                    input = eval(self.hsaffect)(input, cs, 3)
                    out = eval(self.hsmode)(input, 3)
                    inputpj = cv2.cvtColor(input, cv2.COLOR_HSV2RGB)
                    showImage = inputpj
                    showImage = QImage(showImage.data, showImage.shape[1], showImage.shape[0], QImage.Format_RGB888)
                    inputpj = cv2.cvtColor(inputpj, cv2.COLOR_RGB2GRAY)
                    self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
            else:
                if (self.ui.rgb.isChecked()):
                    input = self.I1
                    out = eval(self.hsmode)(input, 1)
                    inputpj = self.I2
                    # cv2.imshow("原图",input)
                    showImage = input
                    showImage = QImage(showImage.data, showImage.shape[1], showImage.shape[0], QImage.Format_RGB888)
                    self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
                elif (self.ui.gray.isChecked()):
                    input = self.I2
                    out = eval(self.hsmode)(input, 2)
                    inputpj = input
                    showImage = input
                    showImage = QImage(showImage.data, showImage.shape[1], showImage.shape[0], QImage.Format_Grayscale8)
                    self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
                else:
                    input = self.I3
                    out = eval(self.hsmode)(input, 3)
                    showImage = self.I1
                    showImage = QImage(showImage.data, showImage.shape[1], showImage.shape[0], QImage.Format_RGB888)
                    inputpj = self.I2
                    self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
            self.pj(inputpj, out)  # GRAY RGB
            outvideo.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            yaresult = QImage(out.data, out.shape[1], out.shape[0], QImage.Format_RGB888)
            self.signals.ya2.emit(QPixmap.fromImage(yaresult), 1)
            # out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("yats.jpg", out)
            # output = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            # cv2.imshow("ya", output)
            # cv2.waitKey(1)
            if self.camclosetag:
                break
        cap.release()
        outvideo.release()
        cv2.destroyAllWindows()
    def handlingthread(self):
        if(self.videotag==True):
            self.yavideo()
            return
        if(self.ui.affect.isChecked()):
            if (self.ui.af1.isChecked()):
                cs = self.fa
                self.hsaffect = "yaway1"
            elif(self.ui.af2.isChecked()):
                cs = self.fb
                self.hsaffect = "yaway2"
            elif(self.ui.af3.isChecked()):
                cs = self.fc
                self.hsaffect = "yaway3"
            else:
                cs = 0
                pass
            if (self.ui.rgb.isChecked()):
                input= self.I1
                input = eval(self.hsaffect)(input, cs,1)
                out = eval(self.hsmode)(input, 1)
                inputpj = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
                showImage = input
                showImage = QImage(showImage.data, showImage.shape[1], showImage.shape[0], QImage.Format_RGB888)
                self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
            elif (self.ui.gray.isChecked()):
                gray = self.I2
                input = eval(self.hsaffect)(gray, cs,2)
                out = eval(self.hsmode)(input, 2)
                inputpj = input
                showImage = input
                showImage = QImage(showImage.data, showImage.shape[1], showImage.shape[0], QImage.Format_Grayscale8)
                self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
            else:
                input = self.I3
                input = eval(self.hsaffect)(input, cs, 3)
                out = eval(self.hsmode)(input, 3)
                inputpj = cv2.cvtColor(input, cv2.COLOR_HSV2RGB)
                showImage = inputpj
                showImage = QImage(showImage.data, showImage.shape[1], showImage.shape[0], QImage.Format_RGB888)
                inputpj = cv2.cvtColor(inputpj, cv2.COLOR_RGB2GRAY)
                self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
        else:
            if (self.ui.rgb.isChecked()):
                input = self.I1
                out = eval(self.hsmode)(input, 1)
                inputpj = self.I2
                #cv2.imshow("原图",input)
                showImage = input
                showImage = QImage(showImage.data, showImage.shape[1], showImage.shape[0], QImage.Format_RGB888)
                self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
            elif (self.ui.gray.isChecked()):
                input = self.I2
                out = eval(self.hsmode)(input, 2)
                inputpj = input
                showImage = input
                showImage = QImage(showImage.data, showImage.shape[1], showImage.shape[0], QImage.Format_Grayscale8)
                self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
            else:
                input = self.I3
                out = eval(self.hsmode)(input, 3)
                showImage = self.I1
                showImage = QImage(showImage.data, showImage.shape[1], showImage.shape[0], QImage.Format_RGB888)
                inputpj = self.I2
                self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
        self.pj(inputpj, out)# GRAY RGB
        yaresult = QImage(out.data, out.shape[1], out.shape[0], QImage.Format_RGB888)
        self.signals.ya2.emit(QPixmap.fromImage(yaresult), 1)
        #out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        #cv2.imwrite("yats.jpg", out)
        return out
    def yaf1(self):
        self.fa = self.ui.fa.value()/1000.0
        self.ui.f1.setText(str(self.fa))
        if (self.ui.zd.isChecked()):
            self.handling()
    def yaf2(self):
        self.fb = self.ui.fb.value() / 1000.0
        self.ui.f2.setText(str(self.fb))
        if (self.ui.zd.isChecked()):
            self.handling()
    def yaf3(self):
        self.fc = self.ui.fc.value() / 1000.0
        self.ui.f3.setText(str(self.fc))
        if (self.ui.zd.isChecked()):
            self.handling()
    def showbusy(self):
        #Ret = MessageBox.question(self.ui, "提醒框", "正在识别中，请耐心等待")  # Critical对话框
        #MessageBox.information(self.ui, "标题", "内容") #！
        self.MessageBox.critical(self.ui, "在忙", "正在识别中，请耐心等待") #err
        #MessageBox.warning(self.ui, "标题", "内容")  #warn

def xc():
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec()
if __name__ == '__main__':
    xc()
    '''
    threadLock = threading.Lock()
    Thread = threading.Thread(target=xc)
    Thread.start()
    '''