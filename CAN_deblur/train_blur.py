#!usr/bin/env python3
#-*- coding=utf-8 -*-
#python=3.6 pytorch=1.2.0


import os
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
from random import randint

from config_blur import cfg, load_cfg
from models.network_factory import get_network
from models.opt_factory import get_opt
from models.loss_factory import get_loss_func
from datasets.loader_factory import get_loader


def print_to_screen(g_loss, d_loss, its, epoch, its_num, writer, logger):
    writer.add_scalars('d_g_loss', {'d loss': d_loss,
                                    'g loss': g_loss}, epoch*its_num+its)
    logger.info(("[%d]{%d}/{%d}"%(epoch, its, its_num)+
        " d_loss:%06f"%(d_loss)+ " g_loss:%06f"%(g_loss)))
        

def save_checkpoints(save_path, model, opt, epoch):
    states = { 'model_state': model.state_dict(),
               'epoch': epoch + 1,
               'opt_state': opt.state_dict(),}
    torch.save(states, save_path)


def load_checkpoints(d_model, g_model, d_opt, g_opt, save_path, logger):
    try:
        states_d = torch.load(save_path.EXPS+save_path.NAME+save_path.DMODEL)
        states_g = torch.load(save_path.EXPS+save_path.NAME+save_path.GMODEL)
        d_model.load_state_dict(states_d['model_state'])
        d_opt.load_state_dict(states_d['opt_state'])
        g_model.load_state_dict(states_g['model_state'])
        g_opt.load_state_dict(states_g['opt_state'])
        current_epoch = states_d['epoch']
        logger.info('loading checkpoints success')
    except:
        current_epoch = 0
        logger.info("no checkpoints")
    return current_epoch


def check_results(g_model, result_path, epoch, img_path):
    try:
        os.mkdir(result_path)
    except:
        pass
    img = cv2.imread(img_path)
    img_line = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
        255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
    img_line = np.expand_dims(img_line, 2)
    img_line = np.expand_dims(img_line, 3)
    img_line_tensor = torch.from_numpy(img_line.astype(np.float32)).permute(3,2,0,1) / 255.

    img_color = img.copy() #with out copy, will change the original img
    for i in range(30):
        randx = randint(0,205)
        randy = randint(0,205)
        img_color[randx:randx+50, randy:randy+50] = 255
        
    img_color = cv2.blur(img_color, (100, 100))
    img_color = np.expand_dims(img_color, 3)
    img_color_tensor = torch.from_numpy(img_color.astype(np.float32)).permute(3,2,0,1) / 255.
    g_model.eval()
    with torch.no_grad():
        output = g_model(torch.cat((img_line_tensor, img_color_tensor),1))
    output = output.squeeze()
    output = output.cpu().numpy().transpose((1,2,0)) * 255
    #output = output.astype(np.uint8)
    cv2.imwrite(result_path+"{}_result.jpg".format(epoch), output)
    g_model.train()


def main():
    logger = load_cfg()     
    torch.manual_seed(cfg.DETERMINISTIC.SEED)
    torch.cuda.manual_seed(cfg.DETERMINISTIC.SEED)
    torch.backends.cudnn.deterministic = cfg.DETERMINISTIC.CUDNN
    np.random.seed(cfg.DETERMINISTIC.SEED)
    #different with common cnn project, no need val data.
    train_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.DATA, 'train', cfg.TRAIN, logger)

    d_model = get_network(cfg.MODEL.DNAME, logger=logger)
    d_model = torch.nn.DataParallel(d_model, cfg.GPUS).cuda()
    g_model = get_network(cfg.MODEL.GNAME, logger=logger)
    g_model = torch.nn.DataParallel(g_model, cfg.GPUS).cuda()
    d_opt, g_opt = get_opt(d_model, g_model, cfg.TRAIN, logger)
    d_loss_func = get_loss_func(cfg.MODEL.DLOSS, logger=logger)
    g_loss_func = get_loss_func(cfg.MODEL.GLOSS, scaling=cfg.TRAIN.L1SCALING, logger=logger)

    current_epoch = load_checkpoints(d_model, g_model, d_opt, g_opt, cfg.PATH , logger)

    log_writter = SummaryWriter(cfg.PATH.EXPS+cfg.PATH.NAME)
    its_num = len(train_loader)
    fake_target = torch.zeros(cfg.TRAIN.BATCHSIZE,1).cuda()
    real_target = torch.ones(cfg.TRAIN.BATCHSIZE,1).cuda()

    for epoch in range(current_epoch, cfg.TRAIN.EPOCHS):
        for its, (sharp_256, sharp_128, sharp_64,blur_256, blur_128, blur_64)in enumerate(train_loader):
            sharp_256 = sharp_256.cuda()
            sharp_128 = sharp_128.cuda()
            sharp_64 = sharp_64.cuda()
            blur_256 = blur_256.cuda()
            blur_128 = blur_128.cuda()
            blur_64 = blur_64.cuda()

            d_opt.zero_grad()
            #logger.debug(blur_64.shape)
            g_results = g_model(blur_256, blur_128, blur_64)
            d_results_fake = d_model(g_results[0])
            d_results_real = d_model(sharp_256) 
            d_loss = d_loss_func(d_results_fake, d_results_real, fake_target, real_target)
            d_loss.backward(retain_graph=True)
            d_opt.step()

            g_opt.zero_grad()
            d_results_fake = d_model(g_results[0])
            g_loss = g_loss_func(g_results, [sharp_256, sharp_128, sharp_64], d_results_fake, real_target)
            g_loss.backward()
            g_opt.step()
            
           # TODO(liu): check it
            # img_test = img_ori[0].cpu().numpy().transpose((1,2,0)) * 255
            # cv2.imwrite("./test.jpg", img_test)
            # cv2.waitKey()
            if its % cfg.PRINT_FRE == 0:
                print_to_screen(g_loss, d_loss, its, epoch, its_num, log_writter, logger)
            
            if cfg.SHORT_TEST == True:
                if its == 20:
                    break
        
        save_checkpoints(cfg.PATH.EXPS+cfg.PATH.NAME+cfg.PATH.DMODEL, d_model, d_opt, epoch)
        save_checkpoints(cfg.PATH.EXPS+cfg.PATH.NAME+cfg.PATH.GMODEL, g_model, g_opt, epoch)
        #check_results(g_model, cfg.PATH.EXPS+cfg.PATH.NAME+cfg.PATH.RESULTS, epoch, cfg.PATH.FIXTEST)
    log_writter.close()


if __name__ == "__main__":
    main()            