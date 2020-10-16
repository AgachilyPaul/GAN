import torch


def get_opt(d_model, g_model, cfg_train, logger=None):
    if cfg_train.D_TYPE == 'adam':
        trainable_vars_d = [param for param in d_model.parameters() if param.requires_grad]
        opt_d = torch.optim.Adam(trainable_vars_d,
            lr=cfg_train.D_LR, 
            betas=(cfg_train.D_BETA1, cfg_train.D_BETA2),
            eps=1e-08, 
            weight_decay=cfg_train.D_WEIGHT_DECAY,
            amsgrad=False)
    else:
        logger.error("{} not exist in opt type".format(cfg_train.D_TYPE))

    if cfg_train.G_TYPE == 'adam':
        trainable_vars_g = [param for param in g_model.parameters() if param.requires_grad]
        opt_g = torch.optim.Adam(trainable_vars_g, 
            lr=cfg_train.G_LR, 
            betas=(cfg_train.G_BETA1, cfg_train.G_BETA2),
            eps=1e-08, 
            weight_decay=cfg_train.G_WEIGHT_DECAY,
            amsgrad=False)
    else:
        logger.error("{} not exist in opt type".format(cfg_train.G_TYPE)) 
    return opt_d , opt_g