import torch
import torch.optim as optim
from torchvision import transforms
from mode.moderrd_net.loss_functions import reconstruction_loss, illumination_smooth_loss, reflectance_smooth_loss, noise_loss, normalize01
import yaconf as conf
import numpy as np
def pipline_retinex(net, img,iterations):
    img_tensor = transforms.ToTensor()(img)  # [c, h, w]
    img_tensor = img_tensor.to(conf.device)
    img_tensor = img_tensor.unsqueeze(0)     # [1, c, h, w]

    optimizer = optim.Adam(net.parameters(), lr=conf.lr)

    # iterations
    for i in range(iterations):
        # forward
        illumination, reflectance, noise = net(img_tensor)  # [1, c, h, w]
        # loss computing
        loss_recons = reconstruction_loss(img_tensor, illumination, reflectance, noise)
        loss_illu = illumination_smooth_loss(img_tensor, illumination)
        loss_reflect = reflectance_smooth_loss(img_tensor, illumination, reflectance)
        loss_noise = noise_loss(img_tensor, illumination, reflectance, noise)

        loss = loss_recons + conf.illu_factor*loss_illu + conf.reflect_factor*loss_reflect + conf.noise_factor*loss_noise

        # backward
        net.zero_grad()
        loss.backward()
        optimizer.step()

    # adjustment
    adjust_illu = torch.pow(illumination, conf.gamma)
    res_image = adjust_illu*((img_tensor-noise)/illumination)
    res_image = torch.clamp(res_image, min=0, max=1)

    if conf.device != 'cpu':
        res_image = res_image.cpu()

    res_img = transforms.ToPILImage()(res_image.squeeze(0))


    return  np.array(res_img)