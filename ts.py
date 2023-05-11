import time

import cv2
import torch.nn as nn
import torch
import numpy as np
import datetime
img0 = cv2.imread("mm.png")
image = cv2.resize(img0, (512, 512))
image = torch.from_numpy(image).float() / 255
image = image.permute(2, 0, 1).unsqueeze(0).cuda()
outa = 2.5
# print(image.shape)
tensor_temp = torch.linspace(0, 1, 256).cuda()


print("优化前(微秒):",end='')
start_time = datetime.datetime.now()
outtemp1 = (image ** outa) * 255
end_time = datetime.datetime.now()
print((end_time-start_time).microseconds)

print("优化后(微秒):",end='')
start_time = datetime.datetime.now()
lut = torch.clamp(255 * torch.pow(tensor_temp, outa), 0, 255).t().cuda()
outtemp2 = lut[(image * 255).type(torch.long)]
end_time = datetime.datetime.now()
print((end_time-start_time).microseconds)

res_img1 = outtemp1.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
res_img1 = res_img1.astype(np.uint8)
cv2.imshow("ts1", res_img1)
res_img2 = outtemp2.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
res_img2 = res_img2.astype(np.uint8)
cv2.imshow("ts2", res_img2)
cv2.waitKey(0)