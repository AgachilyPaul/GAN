import os
import yaml
import logging
from easydict import EasyDict as edict


cfg = edict()
cfg.PATH = edict()
cfg.PATH.DATA = ['./imgs/sharp/','./imgs/blur/'] 
cfg.PATH.TEST = [None,'./imgs/blur/']
cfg.PATH.RES_TEST = './res_imgs/'
cfg.PATH.EXPS = './exps/'
cfg.PATH.NAME = 'blur_v9'
cfg.PATH.DMODEL = '/d_model.pth'
cfg.PATH.GMODEL = '/g_model.pth'
cfg.PATH.LOG = '/log.txt'
cfg.PATH.RESULTS = '/results/'
cfg.PATH.FIXTEST = './imgs/77.jpg'

cfg.DETERMINISTIC = edict()
cfg.DETERMINISTIC.SEED = 0
cfg.DETERMINISTIC.CUDNN = True

cfg.TRAIN = edict()
cfg.TRAIN.EPOCHS = 500
cfg.TRAIN.BATCHSIZE = 2
cfg.TRAIN.L1SCALING = 1e-4
cfg.TRAIN.D_TYPE = 'adam'
cfg.TRAIN.D_LR = 5e-5
cfg.TRAIN.D_BETA1 = 0.9
cfg.TRAIN.D_BETA2 = 0.999
cfg.TRAIN.D_WEIGHT_DECAY = 0
cfg.TRAIN.G_TYPE = 'adam'
cfg.TRAIN.G_LR = 5e-5
cfg.TRAIN.G_BETA1 = 0.9
cfg.TRAIN.G_BETA2 = 0.999
cfg.TRAIN.G_WEIGHT_DECAY = 0
cfg.TRAIN.NUM_WORKERS = 16

cfg.MODEL = edict()
cfg.MODEL.DNAME = 'blur_d' 
cfg.MODEL.GNAME = 'blur_g'
cfg.MODEL.DLOSS = 'bce_d' 
cfg.MODEL.GLOSS = 'MultiBCE_L2_g'

cfg.GPUS = [0]
cfg.PRINT_FRE = 1
cfg.DATASET_TRPE = 'gopro_large'
cfg.SHORT_TEST = False



def load_cfg():
    cfg_name = cfg.PATH.EXPS+cfg.PATH.NAME+'/'+cfg.PATH.NAME+'.yaml'
    if not os.path.exists(cfg.PATH.EXPS+cfg.PATH.NAME):
            os.mkdir(cfg.PATH.EXPS+cfg.PATH.NAME)
    # for log path, can only change by code file
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
        level=logging.DEBUG,
        filename=cfg.PATH.EXPS+cfg.PATH.NAME+cfg.PATH.LOG)
    stream_handler = logging.StreamHandler()
    logger = logging.getLogger(cfg.PATH.NAME)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    if os.path.exists(cfg_name):
        logger.info('start loading config files...')
        seed_add = 10 
        with open(cfg_name) as f:
            old_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
            for k, v in old_cfg.items():
                if k in cfg:
                    if isinstance(v, dict):
                        for vk, vv in v.items():
                            if vk in cfg[k]:
                                cfg[k][vk] = vv
                            else:
                                logger.error("{} not exist in config.py".format(vk))
                    else:
                        cfg[k] = v   
                else:
                   logger.error("{} not exist in config.py".format(k))
        logger.info('loading config files success')
        cfg.DETERMINISTIC.SEED += seed_add
        logger.info('change random seed success')
    else:
        logger.info('start creating config files...')
    cfg_dict = dict(cfg)
    for k, v in cfg_dict.items():
        if isinstance(v, edict):
            cfg_dict[k] = dict(v)
    with open(cfg_name, 'w') as f:
        yaml.dump(dict(cfg_dict), f, default_flow_style=False)
    logger.info('update config files success')
    return logger

if __name__ == "__main__":
    logger = load_cfg()
    print(cfg)




