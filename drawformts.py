import os

import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tabulate import tabulate
import numpy as np
table_header = ['Name', 'Chinese', 'Math', 'English']
list1=['Tom', '90', '80', '85']
list2=['Jim', '70', '90', '80']
table_data = [
       list1,
       list2 ,
       ['Lucy', '90', '70', '90'],
   ]
#print(tabulate(table_data, headers=table_header, tablefmt='presto'))
t=tabulate(table_data, headers=table_header, tablefmt='presto')
#img_source="mm.png"
#img = Image.open(img_source)
line=0
for i in t:
    if(i=='\n'):
        line+=1
fontsize=50
fontlen=len(t.split('\n')[0])+1
img = np.ones([int(fontsize*line*1.2),fontsize*fontlen//2,3], np.uint8)*255
img = Image.fromarray(img)
# 添加文字
draw = ImageDraw.Draw(img)
font = ImageFont.truetype(font=r'D:\zclbs\daima\font\simsun.ttc', size=fontsize)
# 参数：位置、文本、填充、字体
draw.text(xy=(0, 0), text=t, fill=(0, 0, 0), font=font)

cv2.imshow("ya",np.asarray(img))
cv2.waitKey()
#img.show()
#img.save('ts.jpg')
