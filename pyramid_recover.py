import cv2
import matplotlib.pyplot as plt

def gaussian(original_image, down_times):
    temp = original_image.copy()
    gaussian_pyramid = [temp]
    for i in range(down_times):
        temp = cv2.pyrDown(temp)
        #print(temp.shape)
        gaussian_pyramid.append(temp)
    return gaussian_pyramid


def laplacian(gaussian_pyramid, up_times):
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(up_times, 0, -1):
        temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i])
        # temp_lap = cv2.subtract(gaussian_pyramid[i-1], temp_pyrUp)
        temp_lap = gaussian_pyramid[i - 1] - temp_pyrUp
        print(temp_lap.shape)
        laplacian_pyramid.append(temp_lap)
    return laplacian_pyramid

ya1=cv2.imread(r"D:\panload\LOLdataset\our485\low\10.png")
ya2=cv2.imread(r"D:\panload\LOLdataset\our485\high\10.png")
yasize=512
ya1=cv2.resize(ya1,(yasize,yasize))
ya2=cv2.resize(ya2,(yasize,yasize))
#cv2.imshow("ya1b",ya1)
#cv2.imshow("ya2b",ya2)
#cv2.waitKey(0)
deep=5
ya1g=gaussian(ya1,deep)
ya2g=gaussian(ya2,deep)
ya1l=laplacian(ya1g,deep)
ya2l=laplacian(ya2g,deep)

temp=None
def yacover(g,l):
    global temp
    if(l==0):
        temp= ya2l[l+1] + cv2.pyrUp(ya1g[g])
    else:
        if(l<=7):
            temp = ya2l[l + 1] + cv2.pyrUp(temp)
        else:
            temp = ya1l[l+1]+ cv2.pyrUp(temp)
    if(l==deep-1):
        return
    g=g-1
    l=l+1
    return yacover(g,l)
yacover(deep,0)
temp=cv2.resize(temp,(512,512))
cv2.imshow("yat",temp)
cv2.waitKey(0)