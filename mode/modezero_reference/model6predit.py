import torch
import torch.optim
import numpy as np
import torchvision
import cv2

def lowlight(image,DCE_net):
	data_lowlight = image/255.0
	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)
	_,enhanced_image,_ = DCE_net(data_lowlight)

	torchvision.utils.save_image(enhanced_image, "ts.jpg")
	enhanced_image=cv2.imread("ts.jpg")
	enhanced_image= cv2.cvtColor(enhanced_image,cv2.COLOR_BGR2RGB)
	return enhanced_image
	#enhanced_image=enhanced_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

	#return enhanced_image


