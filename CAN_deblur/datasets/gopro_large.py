import os
import numpy as np 
import torch
import torchvision.transforms as transforms 
import cv2
from random import randint

class GOPROLargeData(torch.utils.data.Dataset):
    '''
    return type: tensor 
    return range: 0-1
    '''
    def __init__(self, img_path, aug, test_data=False):
        if img_path[0] is not None:
            self.img_path_sharp = img_path[0]
        self.img_path_blur = img_path[1]
        self.img_list = os.listdir(img_path[1])
        self.test_data = test_data
        self.aug = aug
        self.crop = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop((256,256)),
            transforms.ToTensor()])
            
    def __getitem__(self, index):
        blur_256 = cv2.imread(self.img_path_blur + self.img_list[index])
        if self.test_data == False:
            blur_256 = self.crop(blur_256) 
            blur_256_tensor = blur_256
            blur_128 = blur_256_tensor.clone().numpy()

            blur_128 = blur_128.transpose(1, 2, 0)
            blur_64 = blur_128.copy()

            blur_128 = cv2.resize(blur_128, (128, 128))
            blur_128_tensor = torch.from_numpy(blur_128).permute(2,0,1)

            blur_64 = cv2.resize(blur_64, (64, 64))
            blur_64_tensor = torch.from_numpy(blur_64).permute(2,0,1)
      
            sharp_256 = cv2.imread(self.img_path_sharp + self.img_list[index])
            sharp_256 = self.crop(sharp_256)
           
            sharp_256_tensor = sharp_256
            sharp_128 = sharp_256_tensor.clone().numpy()

            sharp_128 = sharp_128.transpose(1, 2, 0)
            sharp_64 = sharp_128.copy()
            
            sharp_128 = cv2.resize(sharp_128, (128, 128))
            sharp_128_tensor = torch.from_numpy(sharp_128).permute(2,0,1)
            sharp_64 = cv2.resize(sharp_64, (64, 64))
            sharp_64_tensor = torch.from_numpy(sharp_64).permute(2,0,1)

            return sharp_256_tensor, sharp_128_tensor, sharp_64_tensor,\
                blur_256_tensor, blur_128_tensor, blur_64_tensor
        # else: 如果想输入size free 用这段代码
        #     pad_x = 32-blur_256.shape[0]%32
        #     if pad_x == 32: pad_x=0
        #     pad_y = 32-blur_256.shape[1]%32
        #     if pad_y == 32: pad_y=0
        #     img_ori = cv2.resize(blur_256, (blur_256.shape[1]+pad_y,blur_256.shape[0]+pad_x))
        #     img_ori_2 = cv2.resize(img_ori, (img_ori.shape[1]//2, img_ori.shape[0]//2))
        #     img_ori_4 = cv2.resize(img_ori, (img_ori.shape[1]//4, img_ori.shape[0]//4))
            
        #     img_ori_tensor = self.aug(img_ori)
        #     img_ori_2_tensor = self.aug(img_ori_2)
        #     img_ori_4_tensor = self.aug(img_ori_4)
        #     return img_ori_tensor, img_ori_2_tensor, img_ori_4_tensor
        else:
            blur_256 = self.crop(blur_256) 
            blur_256_tensor = blur_256
            blur_128 = blur_256_tensor.clone().numpy()

            blur_128 = blur_128.transpose(1, 2, 0)
            blur_64 = blur_128.copy()

            blur_128 = cv2.resize(blur_128, (128, 128))
            blur_128_tensor = torch.from_numpy(blur_128).permute(2,0,1)

            blur_64 = cv2.resize(blur_64, (64, 64))
            blur_64_tensor = torch.from_numpy(blur_64).permute(2,0,1)
            return blur_256_tensor, blur_128_tensor, blur_64_tensor


    def __len__(self):
        return len(self.img_list)