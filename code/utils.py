""" Utilities """
import os
import logging
import shutil
import glob

import torch
import torchvision.datasets as dset
import numpy as np
import preproc
from configobj import ConfigObj
          

def get_data(dataset, data_path, cutout_length, validation):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10        
    else:    
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    try:
            shape = trn_data.train_data.shape
    except:
            shape = trn_data.data.shape
    input_channels = 3 if len(shape) == 4 else 1
    #assert d shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation: # append validation data
        ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))

    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    os.system('rm '+file_path)
    
    logging.getLogger().handlers = []
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    
    
    return logger



class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        #self.avg = np.nan

def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res

def find_checkpoint(ckpt_dir, seed):
    files = sorted(glob.glob(ckpt_dir+'/'+'checkpoint_{}_*.ckp'.format(seed)), key = lambda x: int(x.split('_')[-1].split('.')[0]))
    if len(files)>0:
        return files[-1], int(files[-1].split('_')[-1].split('.')[0]) + 1
    else:
        return None, 0
    

def save_checkpoint(state, ckpt_dir, seed='', epoch='',  is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint_{}_{}.ckp'.format(seed, epoch))
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best_{}.pth.tar'.format(seed))
        shutil.copyfile(filename, best_filename)


class Config:
    def __init__(self, config_path):            
        self.name = 'experiment'        
        self.workers = 4                         
        self.device = 'cuda:0'
        self.data_path = './data'        
        self.cfg = ConfigObj(config_path)        
        for k in self.cfg:
            self.__dict__[k] = self.cfg[k]
        if 'path' not in self.__dict__:            
            self.path = os.path.join('searchs', self.name)
        self.plot_path = self.path
        
            
        
import time
import torch
import json
import os

class LatencyManager:
    def __init__(self, device, lookup_path="latency_lookup.json"):
        try:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        except Exception:
            self.device = torch.device('cpu')
        self.lookup_path = lookup_path
        self.num_warmup = 5
        self.num_repeats = 20

        if os.path.exists(lookup_path):
            with open(lookup_path, "r") as f:
                self.latency_lookup = json.load(f)
        else:
            self.latency_lookup = {}
    
    def get_latency_key(self, op_name, C_in, C_out, H, W):
        return f"{op_name}___Cin{C_in}___Cout{C_out}___H{H}___W{W}"
    
    def measure_latency(self, op, input_tensor):
        import torch.nn.functional as F
        op.eval()
        
        if input_tensor.shape[0] < 1:
            raise ValueError(f"Input tensor batch size must be >= 1, got {input_tensor.shape[0]}")

        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            for _ in range(self.num_warmup):
                try:
                    _ = op(input_tensor)
                except RuntimeError as e:
                    msg = str(e)
                    if 'Sizes of tensors must match' in msg or 'size mismatch' in msg:
                        _, _, H, W = input_tensor.shape
                        pad_h = (H % 2)
                        pad_w = (W % 2)
                        if pad_h or pad_w:
                            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h))
                            _ = op(input_tensor)
                            break
                    print(f"[ERROR] Warmup failed for {op}: {e}")
                    raise
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            times = []
            for _ in range(self.num_repeats):
                start = time.time()
                _ = op(input_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        return sum(times) / len(times)

    
    def get_op_config(self, op, input_tensor):
        C_in = input_tensor.shape[1]
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        
        C_out = C_in  
        
        if hasattr(op, 'out_channels'):
            C_out = op.out_channels
        elif hasattr(op, 'net'):
            for module in op.net.modules():
                if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'out_channels'):
                    C_out = module.out_channels
                    break
                elif isinstance(module, torch.nn.BatchNorm2d) and hasattr(module, 'num_features'):
                    C_out = module.num_features
                    break
        elif hasattr(op, 'stride'):
            C_out = C_in
        else:
            C_out = C_in
        
        return C_in, C_out, H, W

    
    def get_or_measure_latency(self, op_name, op, input_tensor):
        C_in, C_out, H, W = self.get_op_config(op, input_tensor)
        key = self.get_latency_key(op_name, C_in, C_out, H, W)
        
        if key in self.latency_lookup:
            return self.latency_lookup[key]
        else:
            latency_sec = self.measure_latency(op, input_tensor)
            latency_ms = latency_sec * 1000

            self.latency_lookup[key] = latency_ms
            with open(self.lookup_path, "w") as f:
                json.dump(self.latency_lookup, f, indent=2)
            
            print(f"[LATENCY] Measured {key}: {latency_ms:.3f} ms")
            return latency_ms
    
    def save(self):
        with open(self.lookup_path, "w") as f:
            json.dump(self.latency_lookup, f, indent=2)
