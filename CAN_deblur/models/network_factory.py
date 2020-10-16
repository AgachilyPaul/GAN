
from . import blur_d
from . import blur_g

NET_LUT = {   
        'blur_d': blur_d.Discriminator,
        'blur_g': blur_g.Generator,     
    }

def get_network(net_name, logger=None, cfg=None):
    try:
        net_class = NET_LUT.get(net_name)
    except:
        logger.error("network tpye error, {} not exist".format(net_name))
    net_instance = net_class(cfg=cfg, logger=logger)
    return net_instance
    

if __name__ == "__main__":
    import logging
    from ptflops import get_model_complexity_info
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
        level=logging.DEBUG,)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    model = get_network('blur_d', logger=logger)
    model = model.cuda()
    flops, params = get_model_complexity_info(model,  (3, 256, 256), 
        as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
