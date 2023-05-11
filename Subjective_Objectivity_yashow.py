import math
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import sys
from mode.moderetinex_net.model import RetinexNet
import warnings
import yaconf as conf
from mode.moderrd_net.RRDNet import RRDNet
from mode.moderrd_net.pipline import pipline_retinex
import mode.modezero_reference.model as model6
from mode.modeuretinex_net.test import Inference
from mode.modezero_reference.model6predit import lowlight
from Ya_net import yanet
import threading
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
import os
import cv2
import torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tabulate import tabulate


table_header = ['Algorithm','Mean', 'Std','Entropy', 'PSNR', 'SSIM','Gradient']
warnings.filterwarnings("ignore")
fstr=r"D:\zclbs\daima\data\1.png"
fui=r"GUI\Sub_OBJ.ui"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
Retinex_netpath=conf.Retinex_netpath
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
class yaxc(QObject):
    ya1 = Signal(QPixmap,int)
    ya2 = Signal(QPixmap,int)
    ya3 = Signal(QPixmap, int)
class Stats:
    def __init__(self):
        self.fa = 0
        self.fb = 0
        self.fc = 0
        self.fstr=fstr
        self.ui = QUiLoader().load(fui)
        self.ui.pushButton_5.clicked.connect(self.choose)
        self.ui.pushButton_6.clicked.connect(self.chooseright)
        self.ui.pushButton_7.clicked.connect(self.handling)
        self.signals = yaxc()
        self.signals.ya1.connect(self.pF1)
        self.signals.ya2.connect(self.pF2)
        self.signals.ya3.connect(self.pF3)
        self.ck = QFileDialog(self.ui)
        self.ui.lineEdit_3.setText(self.fstr)
        self.ui.lineEdit_4.setText(self.fstr)
        self.hthr  = threading.Thread(target=self.handlingthread)
        self.MessageBox = QMessageBox()
        #self.ui.img.setScaledContents(True)
        self.ui.img.setFixedSize(1680, 210)
        #self.ui.result.setScaledContents(True)
        self.ui.result1.setFixedSize(1680, 210)
        self.ui.result2.setFixedSize(1680, 210)
        self.I = None

        self.readover()
    def readover(self):  # 显示原图
        self.I = cv2.imread(self.ui.lineEdit_3.text())
        self.I = cv2.resize(self.I, (512, 512))
        img = self.addtext(self.I.copy(),"Input")
        image = np.concatenate([img , img , img , img , img ,img , img, img], axis=1)
        image = cv2.resize(image, (1680, 210))
        showImage = QImage(image.data, image.shape[1], image.shape[0],image.shape[1]*3, QImage.Format_BGR888)
        self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
        self.signals.ya2.emit(QPixmap.fromImage(showImage), 0)
        self.signals.ya3.emit(QPixmap.fromImage(showImage), 0)
    def pF1(self, F1,k):
        if (k == 0):
            self.ui.img.clear()
        else:
            self.ui.img.setPixmap(F1)
    def pF2(self, F2,k):
        if (k == 0):
            self.ui.result1.clear()
        else:
            self.ui.result1.setPixmap(F2)
    def pF3(self, F3,k):
        if (k == 0):
            self.ui.result2.clear()
        else:
            self.ui.result2.setPixmap(F3)
    def choose(self):
        FileDirectory = self.ck.getOpenFileName(self.ui, "请选择要处理的单张图像")
        if (FileDirectory[0] == ''):
            return
        self.ui.lineEdit_3.setText(FileDirectory[0])
        self.readover()
    def chooseright(self):
        FileDirectory = self.ck.getOpenFileName(self.ui, "请选择对应的正确曝光图像")
        if (FileDirectory[0] == ''):
            return
        self.ui.lineEdit_4.setText(FileDirectory[0])
        self.readover()
    def addtext(self,input,text):
        input = cv2.putText(input,text, (0 + 15, 0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)
        input = cv2.copyMakeBorder(input, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return input
    def pjhand(self,inputgray,output,name):
        outputgray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
        lo = yapj1(outputgray)
        yalist=[name,str(lo[0]),str(lo[1]),str(yapj2(outputgray)),str(yapj3(inputgray, outputgray)),str(yapj4(inputgray, outputgray)),str(yapj5(outputgray))]
        return yalist
    def handling(self):
        if(self.hthr.is_alive()==False):
            self.hthr = threading.Thread(target=self.handlingthread)
            self.hthr.start()
        else:
            self.showbusy()
    def handlingthread(self):
        self.signals.ya2.emit(None, 1)
        self.signals.ya3.emit(None, 1)
        image = self.I.copy()
        imageright = cv2.imread(self.ui.lineEdit_4.text())
        imagegray = cv2.cvtColor(imageright, cv2.COLOR_BGR2GRAY)
        imagegray = cv2.resize(imagegray, (512, 512))
        out1 = mode1(image, 1)
        out2 = mode2(image, 1)
        out3 = mode3(image, 1)
        out4 = mode4(image, 1)
        out5 = mode5(image, 1)
        out6 = mode6(image, 1)
        out7 = mode7(image, 1)
        out8 = yamode(image, 1)
        list1 = self.pjhand(imagegray, out1,"Retinex")
        list2 = self.pjhand(imagegray, out2,"Retinex_Net" )
        list3 = self.pjhand(imagegray, out3, "HE")
        list4 = self.pjhand(imagegray, out4, "AHE")
        list5 = self.pjhand(imagegray, out5,  "RRD_Net")
        list6 = self.pjhand(imagegray, out6, "Zero_Reference")
        list7 = self.pjhand(imagegray, out7, "Unfolding_Retinex_Net")
        list8 = self.pjhand(imagegray, out8, "ZCL")
        table_data = [
            list1,list2,list3,list4,list5,list6,list7,list8
        ]
        t = tabulate(table_data, headers=table_header, tablefmt='presto')
        # img_source="mm.png"
        # img = Image.open(img_source)
        line = 0
        for i in t:
            if (i == '\n'):
                line += 1
        fontsize = 50
        fontlen = len(t.split('\n')[0]) + 1
        img = np.ones([int(fontsize * line * 1.2), fontsize * fontlen // 2, 3], np.uint8) * 255
        img = Image.fromarray(img)
        # 添加文字
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font=r'D:\zclbs\daima\font\simsun.ttc', size=fontsize)
        # 参数：位置、文本、填充、字体
        draw.text(xy=(0, 0), text=t, fill=(0, 0, 0), font=font)
        image_ = np.asarray(img)
        cv2.imwrite("objshow.jpg",image_)
        height, width = image_.shape[0],image_.shape[1]
        bl = width  * 1.0 / height
        yaheight = 210
        image_ = cv2.resize(image_, (int(bl * yaheight),yaheight ))
        yaresult = QImage(image_.data, image_.shape[1], image_.shape[0], image_.shape[1] * 3, QImage.Format_BGR888)
        self.signals.ya3.emit(QPixmap.fromImage(yaresult), 1)
        out0 = self.addtext(image, "input")
        out1 = self.addtext(out1, "Retinex")
        out2 = self.addtext(out2, "Retinex_Net")
        out3 = self.addtext(out3, "HE")
        out4 = self.addtext(out4, "AHE")
        out5 = self.addtext(out5, "RRD_Net")
        out6 = self.addtext(out6, "Zero_Reference")
        out7 = self.addtext(out7, "Unfolding_Retinex_Net")
        out8 = self.addtext(out8, "ZCL")
        image = np.concatenate([out1, out2, out3, out4, out5, out6, out7, out8], axis=1)
        cv2.imwrite("subshow.jpg", image)

        image1 = np.concatenate([out0, out1, out2, out4], axis=1)
        image2 = np.concatenate([out5 ,out6, out7, out8], axis=1)
        image3 = np.concatenate([image1, image2], axis=0)
        cv2.imwrite("subshowya.jpg", image3)


        size = 210
        image = cv2.resize(image, (size * 8, size))
        yaresult = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_BGR888)
        self.signals.ya2.emit(QPixmap.fromImage(yaresult), 1)
    def showbusy(self):
        self.MessageBox.critical(self.ui, "在忙", "正在识别中，请耐心等待") #err
def xc():
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec()
if __name__ == '__main__':
    threadLock = threading.Lock()
    xc = threading.Thread(target=  xc())
    xc.start()