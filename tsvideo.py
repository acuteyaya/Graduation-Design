import cv2
from skimage import exposure
import numpy as np
cap = cv2.VideoCapture('data/1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
outvideo = cv2.VideoWriter('output.mp4', fourcc, 60.0, (512,512))

while cap.isOpened():
    ok, frame = cap.read()  # 读取一帧数据
    if not ok:
        break
    frame = cv2.resize(frame, (512, 512))
    rows, cols, channels = frame.shape
    blank = np.ones([rows, cols, channels], frame.dtype)

    cs=0.7
    out = cv2.addWeighted(frame, cs, blank, 1 - cs, 0)

    cs = 1.3
    out = exposure.adjust_gamma(out, cs)  # 调暗
    cv2.imshow("ya", out)
    cv2.waitKey(1)
    outvideo.write(out)
cap.release()
outvideo.release()
cv2.destroyAllWindows()