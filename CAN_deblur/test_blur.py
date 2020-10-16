import os 
import torch 
import torch.nn as nn
import numpy as np
from easydict import EasyDict as edict
import logging
import cv2

from models.network_factory import get_network
from datasets.loader_factory import get_loader
from config_blur import cfg, load_cfg


def load_test_checkpoints(g_model, save_path, logger):
    try:
        #logger.debug(save_path.EXPS+save_path.NAME+save_path.GMODEL)
        states_g = torch.load(save_path.EXPS+save_path.NAME+save_path.GMODEL)
        g_model.load_state_dict(states_g['model_state'])
        logger.info('loading checkpoints success')
    except:
        logger.error("no checkpoints")


def main():
    logger = load_cfg()     
    test_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.TEST, 'test', cfg.TRAIN, logger)
    g_model = get_network(cfg.MODEL.GNAME, logger=logger)
    g_model = torch.nn.DataParallel(g_model, cfg.GPUS).cuda()
    load_test_checkpoints(g_model, cfg.PATH, logger)

    its_num = len(test_loader)
    g_model.eval()
    with torch.no_grad():
        for its, (blur_256_tensor, blur_128_tensor,blur_64_tensor) in enumerate(test_loader):
            blur_256_tensor = blur_256_tensor.cuda()
            blur_128_tensor = blur_128_tensor.cuda()
            blur_64_tensor = blur_64_tensor.cuda()
            g_results = g_model(blur_256_tensor,blur_128_tensor,blur_64_tensor)
            for i in range(blur_256_tensor.shape[0]):
                blur_256_test = blur_256_tensor[i].cpu().numpy().transpose((1,2,0)) * 255
                cv2.imwrite((cfg.PATH.RES_TEST+"line_{}.jpg".format(i+its)), blur_256_test)

                img_res_test = g_results[0][i].cpu().numpy().transpose((1,2,0)) * 255
                cv2.imwrite((cfg.PATH.RES_TEST+"res_{}.jpg".format(i+its)), img_res_test)
                print("{}/{}".format(i+its,its_num))

if __name__ == "__main__":
    main()
    
