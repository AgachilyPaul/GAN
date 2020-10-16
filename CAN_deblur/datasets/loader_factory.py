import numpy as np
import torchvision.transforms as transforms
import torch
import cv2

from . import gopro_large


LOADER_LUT = {   
        'gopro_large' : gopro_large.GOPROLargeData,  
    }


def get_loader(dataset_type, data_path, loader_type, cfg=None, logger=None):
    if loader_type == 'train':
        # augmentation = transforms.Compose([
        #     #transforms.ToPIL(),
        #     #transforms.crop(cfg.)
        #     transforms.ToTensor(),
        # ])
        augmentation = None
        try: 
            _data_class = LOADER_LUT.get(dataset_type)    
        except:
            logger.error("dataset type error, {} not exist".format(dataset_type))

        _data = _data_class(data_path, aug=augmentation) 
        data_loader = torch.utils.data.DataLoader(_data,
        batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
        drop_last=True)
        
    elif loader_type == 'self_test':
        # augmentation = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        augmentation = None
        try: 
            _data_class = LOADER_LUT.get(dataset_type)    
        except:
            logger.error("dataset type error, {} not exist".format(dataset_type))
        _data = _data_class(data_path, aug=augmentation) 
        data_loader = torch.utils.data.DataLoader(_data,
        batch_size=2, shuffle=True, num_workers=0,
        drop_last=True)
        
    elif loader_type == 'test':
        augmentaiton = transforms.Compose([
            transforms.ToTensor()
        ])
        try: 
            _data_class = LOADER_LUT.get(dataset_type)    
        except:
            logger.error("dataset type error, {} not exist".format(dataset_type))
        _data = _data_class(data_path, aug=augmentaiton, test_data=True)
        data_loader = torch.utils.data.DataLoader(_data,
        batch_size=1, shuffle=False, num_workers=0,
        drop_last=False)

    else:
        logger.error("error, only support train type dataloader")
    
    return data_loader

 
def inverse_preprocess(image):
    image = image.numpy().transpose((1,2,0)) * 255
    image = image.astype(np.uint8)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        #pass
    return image


def test_gopro():
    import matplotlib.pyplot as plt 
    import random
    
    
    random.seed(0)
    torch.manual_seed(0)
    
    train_loader = get_loader("gopro_large", 
        ['/home/liuhaiyang/dataset/GOPR0378_13_00/sharp/','/home/liuhaiyang/dataset/GOPR0378_13_00/blur/'],
        "self_test")
    for i, (sharp_256_tensor, sharp_128_tensor, sharp_64_tensor,blur_256_tensor, blur_128_tensor,blur_64_tensor) in enumerate(train_loader):
        sharp_256_tensor = inverse_preprocess(sharp_256_tensor[0])
        sharp_128_tensor = inverse_preprocess(sharp_128_tensor[0])
        sharp_64_tensor = inverse_preprocess(sharp_64_tensor[0])
        blur_256_tensor = inverse_preprocess(blur_256_tensor[0])
        blur_128_tensor = inverse_preprocess(blur_128_tensor[0])
        blur_64_tensor = inverse_preprocess(blur_64_tensor[0])
        # img_ori_2 = inverse_preprocess(img_ori_tensor[1])
        # img_line_2 = inverse_preprocess(img_line_tensor[1])
        # img_color_2 = inverse_preprocess(img_color_tensor[1])
        
        fig = plt.figure()
        a = fig.add_subplot(2,3,1)
        a.set_title('img_ori_1')
        plt.imshow(sharp_256_tensor)
        a = fig.add_subplot(2,3,2)
        a.set_title('sharp_128_tensor')
        plt.imshow(sharp_128_tensor)
        a = fig.add_subplot(2,3,3)
        a.set_title('img_color_1')
        plt.imshow(sharp_64_tensor)

        a = fig.add_subplot(2,3,4)
        a.set_title('img_ori_2')
        plt.imshow(blur_256_tensor)
        a = fig.add_subplot(2,3,5)
        a.set_title('img_line_2')
        plt.imshow(blur_128_tensor)
        a = fig.add_subplot(2,3,6)
        a.set_title('img_color_2')
        plt.imshow(blur_64_tensor)
        plt.show()


if __name__ == "__main__":
    test_gopro()