""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
np.random.seed(0)
import sys
from tensorboardX import SummaryWriter
import utils
from importlib import import_module
from utils import Config
from tqdm import tqdm
def main(path_to_cfg):    
    config = Config(path_to_cfg)
    device = config.device

    es_patience = int(getattr(config, 'early_stopping_patience', 0))   # 0 => disabled
    es_min_delta = float(getattr(config, 'early_stopping_min_delta', 0.0))
    es_warmup = int(getattr(config, 'early_stopping_warmup', 0))       # epochs before monitoring
    # mode: 'auto'|'min'|'max' — если не задан, определим ниже от выбранной метрики
    es_mode = getattr(config, 'early_stopping_mode', 'auto')

    # tensorboard
    tb_path = os.path.join(config.path, "tb")
    os.system('rm -r '+tb_path)
    writer = SummaryWriter(log_dir=tb_path)
    writer.add_text('config', str(config.__dict__), 0)

    logger = utils.get_logger(os.path.join(
        config.path, "{}_train.log".format(config.name)))    

    logger.info("Logger is set - training start")
    torch.cuda.set_device(config.device)


    torch.backends.cudnn.benchmark = True

    # get data with meta info
    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=int(config.cutout), validation=True)
    
    module_name, class_name = config.model_class.rsplit('.', 1)
    controller_cls = getattr(import_module(module_name), class_name)
    model = None 
    for seed in config.seeds.split(';'):    
        seed = int(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # deleting model from previous seed
        if model is not None:
            del model 
            if 'cuda' in config.device:
                torch.cuda.empty_cache()        
        model_params = config.__dict__
        model_params['seed'] = seed
        model = controller_cls(**model_params)
        # model.log = logger
        model = model.to(device)    
        # split data to train/validation
        n_train = len(train_data)
        
        if float(config.validate_split)>=0:
            split = int(n_train * float(config.validate_split))
            indices = list(range(n_train))
            if split <= 0:
                logger.info('using train as validation')
                valid_sampler = train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    indices)
            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    indices[:split])
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    indices[split:])
                train_loader = torch.utils.data.DataLoader(train_data,batch_size=int(config.batch_size), sampler=train_sampler,  num_workers=int(config.workers),    pin_memory=True)
                valid_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=int(config.batch_size),
                                                sampler=valid_sampler,
                                                num_workers=int(config.workers),
                                                pin_memory=True)

        
        else:
            logger.info('using test as validation. Use it only for the training with already defined architecture!')
            train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=int(config.batch_size),
                                               shuffle=True,
                                               num_workers=int(config.workers),
                                               pin_memory=True)
            valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=int(config.batch_size),
                                               shuffle=False,
                                               num_workers=int(config.workers),
                                               pin_memory=True)
        # training loop
        # best = 0 
        best = None
        no_improve_epochs = 0  

        if es_mode == 'auto':
            cmp_mode = 'max'
        else:
            cmp_mode = es_mode
        # trying to find available models
        checkpoint_path, start_epoch = utils.find_checkpoint(config.path, seed)
        if checkpoint_path is not None:
            logger.debug('found checkpoint: '+checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path))


        

        for epoch in range(start_epoch, int(config.epochs)):
            # it's better to control seed if ti changes during every epoch                        
            logger.info('epoch' +str(epoch))
            logger.info('using seed '+str(seed+epoch))
            np.random.seed(seed+epoch)
            torch.manual_seed(seed+epoch)
            torch.cuda.manual_seed_all(seed+epoch)

            model.new_epoch(epoch, writer, logger) 
            # training
            if float(config.validate_split) <= 0.0:
                val_train_loader = train_loader
            else:
                val_train_loader = valid_loader
 
            train_qual = train(train_loader, val_train_loader, model, epoch, writer,  config, logger)        
                        
            # validation
            cur_step = (epoch+1) * len(train_loader)
            val_qual = validate(valid_loader, model, epoch,
                                cur_step, device, config, logger, writer)
            plot_path = os.path.join(config.plot_path, "EP{:02d}_Seed{}".format(epoch+1, seed))                            
            caption = "Epoch {}".format(epoch+1)
            #model.plot_genotype(plot_path, caption)
            
            if int(config.use_train_quality) != 0:
                cur_qual = train_qual
            else:
                cur_qual = val_qual

            is_best = False
            if best is None:
                best = cur_qual
                is_best = True
                no_improve_epochs = 0
            else:
                improvement = (cur_qual - best) if cmp_mode == 'max' else (best - cur_qual)
                if improvement > es_min_delta:
                    best = cur_qual
                    is_best = True
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

            if is_best or epoch % int(config.save_every) == 0:
                 utils.save_checkpoint(model.state_dict(), config.path, seed, epoch, is_best=is_best)
            logger.info("Quality{}: {} \n\n".format(
                '*' if is_best else '', cur_qual))
            # tensorboard & logs for ES
            try:
                writer.add_scalar('early_stop/no_improve_epochs', no_improve_epochs, (epoch+1) * len(train_loader))
                writer.add_scalar('early_stop/best', best, (epoch+1) * len(train_loader))
            except Exception:
                pass

            # --- Early Stopping trigger ---
            if es_patience > 0 and epoch + 1 >= es_warmup and no_improve_epochs >= es_patience:
                logger.info(f"Early stopping at epoch {epoch+1}: "
                            f"no improvement > {es_min_delta} for {no_improve_epochs} epoch(s). "
                            f"Best={best}")
                break

        logger.info("Final best =  {}".format(best))    


def train(train_loader, valid_loader, model, epoch, writer,  config, logger):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    cur_step = epoch*len(train_loader)
    
    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in (enumerate(zip(train_loader, valid_loader))):
        trn_X, trn_y = trn_X.to(config.device, non_blocking=True), trn_y.to(
            config.device, non_blocking=True)
        val_X, val_y = val_X.to(config.device, non_blocking=True), val_y.to(
            config.device, non_blocking=True)
        N = trn_X.size(0)

        loss = model.train_step(trn_X, trn_y, val_X, val_y)
        losses.update(loss.item(), N)

        if  step ==  len(train_loader)-1:
            model.eval()
            logits = model(trn_X)
            model.train()
            prec1 = utils.accuracy(
                logits, trn_y)[0]
            top1.update(prec1.item(), N)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Acc@(1) {top1.avg:.1%} | Params: {total_params:,}".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, 
                    losses=losses, top1=top1, total_params=total_params)
            )
        
        
            writer.add_scalar('train/loss', loss.item(), cur_step)
            writer.add_scalar('train/top1', prec1.item(), cur_step)
            
        cur_step += 1

    logger.info(
        "Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    logger.info(
        "Train: [{:2d}/{}] Final Loss {:.4}".format(epoch+1, config.epochs, losses.avg))
    if config.quality == 'negloss':
        return -losses.avg
    elif config.quality == 'top1':
        return top1.avg
    elif config.quality == 'last':
        return cur_step


def validate(valid_loader, model, epoch, cur_step, device, config, logger, writer):
    top1 = utils.AverageMeter()    
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(
                device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)
            prec1 = utils.accuracy(
                logits, y)[0]

            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
                            

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/top1', top1.avg, cur_step)
    

    logger.info(
        "Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))
    logger.info(
        "Valid: [{:2d}/{}] Final Loss {:.4}".format(epoch+1, config.epochs, losses.avg))

    if config.quality == 'negloss':
        return -losses.avg
    elif config.quality == 'top1':
        return top1.avg
    elif config.quality == 'last':
        return cur_step


if __name__ == "__main__":
    main(path_to_cfg = 'configs/my_configs/fmnist_fine_0.cfg' )
    # main(path_to_cfg = 'configs/my_cifar/cifar_fine_0.cfg' )
    # main(path_to_cfg = 'configs/fmnist_5_ops/kappa_50.cfg' )
    # main(path_to_cfg = 'configs/my_configs/kappa_50_3_ops_more_steps.cfg' )
    # {'avg_pool_3x3': 0, 'skip_connect': 1, 'sep_conv_3x3': 2}