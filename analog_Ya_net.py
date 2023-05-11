import threading
import time

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import torch.fft as fft
import yaconf

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
fstr=r"D:\panload\LOLdataset\eval15\low\547.png"
fui=r"GUI\ts.ui"
class yaxc(QObject):
    ya1 = Signal(QPixmap,int)
    ya2 = Signal(QPixmap,int)
class Stats:
    def __init__(self):
        self.fa = 0
        self.fb = 0
        self.fc = 0
        self.fstr=fstr
        self.ui = QUiLoader().load(fui)
        self.ui.pushButton_5.clicked.connect(self.choose)
        self.ui.pushButton_6.clicked.connect(self.handling)
        self.signals = yaxc()
        self.signals.ya1.connect(self.pF1)
        self.signals.ya2.connect(self.pF2)
        self.ck = QFileDialog(self.ui)
        self.ui.lineEdit_3.setText(self.fstr)
        self.hthr  = threading.Thread(target=self.handlingthread)
        self.MessageBox = QMessageBox()
        self.ui.img.setScaledContents(True)
        self.ui.img.setFixedSize(600, 600)
        self.ui.result.setScaledContents(True)
        self.ui.result.setFixedSize(600, 600)
        self.ui.zd.clicked.connect(self.zdhandling)
        self.I = None
        self.ui.fa.setRange(0, 1000)
        self.ui.fa.setValue(500)
        self.ui.fb.setRange(0, 1000)
        self.ui.fb.setValue(500)
        self.ui.fc.setRange(0, 1000)
        self.ui.fc.setValue(500)
        self.ui.f1.setText(str(round(yaconf.yap(self.ui.fa.value() / 1000.0, 1),2)))
        self.ui.f2.setText(str(round(yaconf.yap(self.ui.fb.value() / 1000.0, 2),2)))
        self.ui.f3.setText(str(round(yaconf.yap(self.ui.fc.value() / 1000.0, 3),2)))
        self.ui.fa.sliderReleased.connect(self.yaf1)
        self.ui.fb.sliderReleased.connect(self.yaf2)
        self.ui.fc.sliderReleased.connect(self.yaf3)
        self.ui.af1.clicked.connect(self.af123)
        self.ui.af2.clicked.connect(self.af123)
        self.ui.af3.clicked.connect(self.af123)
        self.ui.af1.setChecked(True)
        self.ui.af2.setChecked(True)
        self.ui.af3.setChecked(True)
        self.ui.zd.setChecked(True)
        self.readover()
    def af123(self):
        if (self.ui.af1.isChecked()):
            self.ui.f1.setEnabled(True)
            self.ui.fa.setEnabled(True)
        else:
            self.ui.f1.setEnabled(False)
            self.ui.fa.setEnabled(False)
        if(self.ui.af2.isChecked()):
            self.ui.f2.setEnabled(True)
            self.ui.fb.setEnabled(True)
        else:
            self.ui.f2.setEnabled(False)
            self.ui.fb.setEnabled(False)
        if (self.ui.af3.isChecked()):
            self.ui.f3.setEnabled(True)
            self.ui.fc.setEnabled(True)
        else:
            self.ui.f3.setEnabled(False)
            self.ui.fc.setEnabled(False)

    def yaf1(self):
        self.fa = round(yaconf.yap(self.ui.fa.value() / 1000.0, 1),2)
        self.ui.f1.setText(str(self.fa))
        if (self.ui.zd.isChecked()):
            self.handling()
    def yaf2(self):
        self.fb = round(yaconf.yap(self.ui.fb.value() / 1000.0, 2),2)
        self.ui.f2.setText(str(self.fb))
        if (self.ui.zd.isChecked()):
            self.handling()
    def yaf3(self):
        self.fc = round(yaconf.yap(self.ui.fc.value() / 1000.0, 3),2)
        self.ui.f3.setText(str(self.fc))
        if (self.ui.zd.isChecked()):
            self.handling()
    def zdhandling(self):
        if (self.ui.zd.isChecked()):
            self.handling()
    def readover(self):  # 显示原图
        self.I = cv2.imread(self.ui.lineEdit_3.text())
        #cv2.imshow("ya",self.I)
        #cv2.waitKey(0)
        #self.I = cv2.resize(self.I,(320,320))
        input = self.I
        showImage = QImage(input.data, input.shape[1], input.shape[0],input.shape[1]*3, QImage.Format_BGR888)
        self.signals.ya1.emit(QPixmap.fromImage(showImage), 1)
        self.signals.ya2.emit(QPixmap.fromImage(showImage), 0)
        if (self.ui.zd.isChecked()):
            self.handling()
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
    def choose(self):
        FileDirectory = self.ck.getOpenFileName(self.ui, "请选择要处理的单张图像")
        if (FileDirectory[0] == ''):
            return
        self.ui.lineEdit_3.setText(FileDirectory[0])
        self.readover()
    def handling(self):
        if(self.hthr.is_alive()==False):
            self.hthr = threading.Thread(target=self.handlingthread)
            self.hthr.start()

        else:
            self.showbusy()

    def handlingthread(self):
        #self.signals.ya2.emit(None, 1)
        img0 = self.I.copy()
        image = cv2.resize(img0, (512, 512))
        image = torch.from_numpy(image).float() / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        outa = yaconf.yap(self.ui.fa.value() / 1000.0, 1)
        outb = yaconf.yap(self.ui.fb.value() / 1000.0, 2)
        outc = yaconf.yap(self.ui.fc.value() / 1000.0, 3)
        #print(image.shape)
        Tersor_one = torch.ones([1]).cuda()
        lut = torch.clamp(255 * torch.pow(torch.linspace(0, 1, 256).cuda(), outa), 0, 255).t()
        outtemp = lut[(image * 255).type(torch.long)]
        at = Tersor_one * outb
        bt = Tersor_one * 255 * outc
        out_contrast = torch.clamp((at * outtemp + bt), 0, 255)
        out2 = out_contrast / 255.0

        h,w=512,512
        lpf = torch.zeros((h, w))
        R = 50  # 或其他
        for x in range(w):
            for y in range(h):
                if ((x - (w - 1) / 2) ** 2 + (y - (h - 1) / 2) ** 2) < (R ** 2):
                    lpf[y, x] = 1
        hpf = 1 - lpf
        hpf, lpf = hpf.cuda(), lpf.cuda()

        f1 = fft.fftn(out2, dim=(2, 3))
        f = torch.roll(f1, (h // 2, w // 2), dims=(2, 3))  # 移频操作,把低频放到中央
        f_l = f * lpf
        f_h = f * hpf
        X_l = torch.abs(fft.ifftn(f_l, dim=(2, 3)))
        X_h = torch.abs(fft.ifftn(f_h, dim=(2, 3)))

        X= (X_h * 255).type(torch.uint8)
        #print(X.shape)
        Y=torch.nonzero(X>100,as_tuple=False)
        #print(Y.shape[0])
        res_img = out2.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0)) * 255
        res_img = res_img.astype(np.uint8)
        showImage = QImage(res_img.data, res_img.shape[1], res_img.shape[0], res_img.shape[1] * 3, QImage.Format_BGR888)
        self.signals.ya2.emit(QPixmap.fromImage(showImage), 1)
        res_img = X_h.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0)) * 255
        res_img = res_img.astype(np.uint8)
        cv2.imwrite("ts.jpg",res_img)
        #time.sleep(3)
    def showbusy(self):
        #Ret = MessageBox.question(self.ui, "提醒框", "正在识别中，请耐心等待")  # Critical对话框
        #MessageBox.information(self.ui, "标题", "内容") #！
        self.MessageBox.critical(self.ui, "在忙", "正在识别中，请耐心等待") #err
        #MessageBox.warning(self.ui, "标题", "内容")  #warn
        #print(Ret)
def xc():
    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec()
def yashow():
    while(1):
        threadLock.acquire()
        res = cv2.imread("ts.jpg")
        threadLock.release()
        try:
            cv2.imshow("ts", res)
            cv2.waitKey(1)
        except:
            continue
if __name__ == '__main__':
    threadLock = threading.Lock()
    show = threading.Thread(target=yashow)
    show.start()
    xc = threading.Thread(target=  xc())
    xc.start()

