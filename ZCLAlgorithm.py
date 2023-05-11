import glob
import numpy as np
import cv2

class yaclass():
    def __init__(self):
        self.datatur=0
        self.fn  = []
        self.fla = []
        self.flb = []
        self.fha = []
        self.fhb = []
        self.fi = []
        self.lolfl=[]
        self.lolfh=[]
    def maketxt(self,txtpath,yalist):
        file = open(txtpath, 'w')
        for data in yalist:
            file.write(data + "\n")
        file.close()
    def readtxt(self, txtpath):
        yalist = []
        file = open(txtpath, 'r')
        for data in file.readlines():
            yalist.append(data.strip('\n'))
        file.close()
        return yalist
    def yamakelol(self,path=r'D:\panload\LOLdataset'):
        self.lolfl = []
        self.lolfh = []
        ftr = glob.glob(path + "/*485")
        fte = glob.glob(path + "/*15")
        fltr=glob.glob(ftr[0] + "//low"+'/*.png')
        fhtr=glob.glob(ftr[0] + "//high"+'/*.png')
        flte = glob.glob(fte[0] + "//low" + '/*.png')
        fhte = glob.glob(fte[0] + "//high" + '/*.png')
        self.lolfl = fltr
        self.lolfh = fhtr
        self.maketxt('flolldata', self.lolfl)
        self.maketxt('flolhdata', self.lolfh)
        self.datatur=(fltr,fhtr,flte,fhte)
    def yaloadlol(self):
        self.lolfl = self.readtxt('flolldata')
        self.lolfh = self.readtxt('flolhdata')
    def yamakekinds(self,path=r'D:\data\training\INPUT_IMAGES'):
        self.fn=[]
        self.fla=[]
        self.flb=[]
        self.fha=[]
        self.fhb=[]
        for i in range(1, 4501):
            f="a{}".format(('%04d' % i))
            f = glob.glob(path + "\\" + f+'*.JPG')
            if(len(f)==0):
                continue
            self.fn.append(f[0])
            self.fla.append(f[1])
            self.flb.append(f[2])
            self.fha.append(f[3])
            self.fhb.append(f[4])
        self.maketxt('fndata',self.fn)
        self.maketxt('fladata',self.fla)
        self.maketxt('flbdata', self.flb)
        self.maketxt('fhadata', self.fha)
        self.maketxt('fhbdata', self.fhb)
        print("保存文件成功，处理结束")
    def yamakefi(self, path=r'D:\data\training\GT_IMAGES'):
        self.fi = []
        f = glob.glob(path + "/*.jpg")
        for i in f:
            self.fi.append(i)
        self.maketxt('fidata', self.fi)
    def yaloadkinds(self):
        self.fi = self.readtxt('fidata')
        self.fn = self.readtxt('fndata')
        self.fla = self.readtxt('fladata')
        self.flb = self.readtxt('flbdata')
        self.fha = self.readtxt('fhadata')
        self.fhb = self.readtxt('fhbdata')
        self.yashow()
    def yashow(self):
        #heigh=15
        #width=20
        heigh=120
        width=120
        xlen=3
        ylen=15
        imagex = np.zeros([heigh, width, 3]).astype(np.uint8)
        imagey = np.zeros([heigh, width*xlen*6, 3]).astype(np.uint8)
        j=0
        k=0
        firsttagx = True
        firsttagy = True
        for i in range(len(self.fn)):
            imgfi = cv2.imread(self.fi[i])
            imgfn=cv2.imread(self.fn[i])
            imgfla = cv2.imread(self.fla[i])
            imgflb = cv2.imread(self.flb[i])
            imgfha = cv2.imread(self.fha[i])
            imgfhb = cv2.imread(self.fhb[i])
            imgfi = cv2.resize(imgfi, (width, heigh))
            imgfn  = cv2.resize(imgfn,  (width, heigh))
            imgfla = cv2.resize(imgfla, (width, heigh))
            imgflb = cv2.resize(imgflb, (width, heigh))
            imgfha = cv2.resize(imgfha, (width, heigh))
            imgfhb = cv2.resize(imgfhb, (width, heigh))
            imaget = np.concatenate([imgfi,imgfla,imgflb,imgfn,imgfhb,imgfha], axis=1)
            k+=1
            if(k==xlen+1):
                k=0
                j+=1
                if (j == ylen + 1):
                    j = 0
                    #cv2.imshow("zcl", imagey)
                    cv2.imwrite("ya.png",imagey)
                    print("ok")
                    return
                    imagey = np.zeros([heigh, width*xlen*6, 3]).astype(np.uint8)
                    firsttagx = True
                    firsttagy = True
                    imagex = np.zeros([heigh, width, 3]).astype(np.uint8)
                else:
                    imagey = np.concatenate([imagey, imagex], axis=0)
                    if (firsttagy):
                        imagey = np.delete(imagey, slice(0, heigh), 0)
                        firsttagy = False
                    firsttagx = True
                    imagex = np.zeros([heigh, width, 3]).astype(np.uint8)
            else:
                imagex = np.concatenate([imagex, imaget], axis=1)
                if (firsttagx):
                    imagex = np.delete(imagex, slice(0, width), 1)
                    firsttagx = False
if __name__ == '__main__':
    c = yaclass()
    #c.yamakelol()
    c.yaloadkinds()
