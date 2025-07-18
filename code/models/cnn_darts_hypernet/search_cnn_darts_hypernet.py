from ..cnn.search_cnn import SearchCNNController, SearchCNN
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from models.cnn.search_cells import SearchCell
from models.cnn_darts_hypernet.architect import HypernetArchitect
from models.cnn import ops
from torch.nn.parallel._functions import Broadcast
from visualize import plot
import genotypes as gt
import logging
import numpy as np
from utils import get_latency_table
import torch.nn.init as init
SIMPLEX_D = 3

class Hypernet2(nn.Module):
    def __init__(self, size, kernel_num):  
        nn.Module.__init__(self)    
        self.size = size
        if isinstance(size, int):
            self.total_size = size
        else:
            self.total_size = np.prod(size)
        
        self.arch_references = nn.Parameter(torch.randn(kernel_num, self.total_size))
        self.pivots = nn.Parameter(torch.randn(kernel_num, SIMPLEX_D))
        
        
                    
    def forward(self, lam): 
        dists = torch.softmax(((lam - self.pivots)**2).sum(1), 0)
        result = dists.view(1, -1) @ self.arch_references
        if isinstance(self.size, int):
            return result
            
        return result.view(*self.size)
        
        
class HyperLinear(nn.Module):
    def __init__(self, size, kernel_num):
        nn.Module.__init__(self)
        self.weight = Hypernet2(size, kernel_num)
        self.bias = Hypernet2(size[1], kernel_num)  # size[1] is the output dimension, там перепутано в  SearchCNNWithHyperNet self.linear
    
    def forward(self, x, lam):
        weight = self.weight(lam).float()
        bias = self.bias(lam).float()
        res =  torch.matmul(x, weight) + bias         
        return res
        
# class PWNet(nn.Module):
#     def __init__(self, size, kernel_num,  init_ = 'random'):    
#         nn.Module.__init__(self)
        
#         if not isinstance(size, tuple): # check if size is 1d
#             size = (size,)
            
#         self.size = size
        
        
#         full_param_size = np.prod(self.size)        
#         total_size = [kernel_num]+list(self.size)
#         self.kernel_num = kernel_num  
        
#         #self.const = nn.Parameter(t.randn(size))
        
        
#         self.const = nn.Parameter(torch.randn(total_size, dtype=torch.float32))
#         if init_ == 'random':
            
#             for i in range(kernel_num):
#                 if len(self.size)>1:
#                     init.kaiming_uniform_(self.const.data[i], a= np.sqrt(5))
#                 else:
                
#                     self.const.data[i]*=0
#                     self.const.data[i]+=torch.randn(size)
                
#         else:
#             self.const.data *=0
#             self.const.data += init_              
#         self.pivots = nn.Parameter(torch.tensor(np.linspace(0, 1,kernel_num)), requires_grad=True)
        
            
#     def forward(self, lam):   
#         lam_ = lam * 0.99999
#         left = torch.floor(lam_*(self.kernel_num-1)).long() 
#         right = left + 1      
#         dist = (self.pivots[right]-lam_)/(self.pivots[right]-self.pivots[left])
#         res = self.const[left] * (dist) + (1.0-dist) * self.const[right]
        
#         return res


# class PWLinear(nn.Module):
#     def __init__(self, size, kernel_num):
#         nn.Module.__init__(self)
#         self.weight = PWNet(size, kernel_num)
#         self.bias = PWNet(size[1], kernel_num)
    
#     def forward(self, x, lam):
#         weight = self.weight(lam).float()
#         bias = self.bias(lam).float()
#         res =  torch.matmul(x, weight) + bias         
#         return res
        

class HyperNet(nn.Module):
    """
    гиперсеть, управляющая нашей структурой
    """
    def __init__(self, hidden_layer_num, hidden_size, out_size1, out_size2, kernel_num):
        """
        :param hidden_layer_num: количество скрытых слоев (может быть нулевым)
        :param hidden_size: количество нейронов на скрытом слое (актуально, если скрытые слои есть)
        :param out_size1: количество строк в матрице, задающей структуру
        :param out_size2: количество столбцов в матрице, задающей структуру
        """
        nn.Module.__init__(self)
        self.out_size1 = out_size1
        self.out_size2 = out_size2
        out_size = out_size1 * out_size2 # выход MLP - вектор, поэтому приводим матрицу к вектору
        layers = []
        
                
        if hidden_layer_num > 0:
            raise NotImplementedError('deprecated?')
        
        self.model = nn.Sequential(Hypernet2(out_size, kernel_num))
       

    def forward(self, x):
        # x --- одномерный вектор (задающий сложность)        
        res =  self.model(x).view(self.out_size1, self.out_size2).float()        
        return res

    
    
class SearchCNNWithHyperNet(SearchCNN):
    def __init__(self, kernel_num, primitives,  C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        SearchCNN.__init__(self, primitives,  C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)
        #self.linear = Hypernet2(, kernel_num)
        self.linear = HyperLinear((self.linear.weight.shape[1], self.linear.weight.shape[0]), kernel_num)  # перепутан порядок, надо [0],[1] для понятности
        
        
    
    def forward(self, x, lam, weights_normal, weights_reduce):
        
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out, lam).view(out.shape[0], -1)
        #logits = self.linear(out)
        
        return logits
    
class SearchCNNControllerWithHyperNet(SearchCNNController):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        subcfg = kwargs['darts']
        C_in = int(subcfg['input_channels'])
        C = int(subcfg['init_channels'])
        n_classes = int(subcfg['n_classes'])
        n_layers = int(subcfg['layers'])
        n_nodes = int(subcfg['n_nodes'])
        stem_multiplier = int(subcfg['stem_multiplier'])
        self.sampling_mode = subcfg['sampling_mode']
        self.n_nodes = n_nodes
        self.device = kwargs['device']
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.t = float(subcfg['initial temp'])
        self.init_t = float(subcfg['initial temp'])

        self.delta_t = float(subcfg['delta'])
        primitives = self.get_primitives(kwargs)
        global SIMPLEX_D 
        SIMPLEX_D = len(primitives)
        self.connect_dict = {elem: i for i, elem in enumerate(primitives)}
        print(self.connect_dict)
        # self.logger = None

        self.lam_log_min = float(kwargs['hypernetwork']['log10_lambda_min']) # логарифм минимально допустимой лямбды
        self.lam_log_max = float(kwargs['hypernetwork']['log10_lambda_max']) # логарифм максимально допустимой лямбды
        
        self.latency_kappa = float(kwargs['latency']['kappa'])

        self.kernel_num = int(self.lam_log_max - self.lam_log_min) + 1
        
        self.init_alphas(kwargs)
        self.net = SearchCNNWithHyperNet(self.kernel_num, primitives, C_in, C, n_classes, n_layers,
                             n_nodes, stem_multiplier)
        ##########
        # return of 2 optimizers
        self.w_optim = torch.optim.SGD(self.weights(), 
                                         lr=float(subcfg['optim']['w_lr']), 
                                         momentum=float(subcfg['optim']['w_momentum']),
                                         weight_decay=float(subcfg['optim']['w_weight_decay']))
        self.alpha_optim = torch.optim.Adam(self.alphas(), 
                                             lr=float(subcfg['optim']['alpha_lr']), 
                                             betas=(0.5, 0.999),
                                             weight_decay=float(subcfg['optim']['alpha_weight_decay']))
                                             
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.w_optim, int(kwargs['epochs']), eta_min=float(subcfg['optim']['w_lr_min']))

        self.simple_alpha_update = int(subcfg['optim']['simple_alpha_update']) != 0
        self.w_grad_clip = float(subcfg['optim']['w_grad_clip'])
        self.lam_sample_num = int(kwargs['hypernetwork']['lambda sample num'])
        if not self.simple_alpha_update and self.lam_sample_num > 1:
            raise NotImplementedError('Bad sample num with advanced alpha optimization')
            
        ##########
        self.cur_e = 0
        self.epochs = int(kwargs['epochs'])
        
        

    def init_alphas(self,  config):
        primitives = self.get_primitives(config)
        subcfg = config['darts']
        n_layers = int(subcfg['layers'])
        n_nodes = int(subcfg['n_nodes'])
        
        hypernetwork_hidden_layer_num = int(
            config['hypernetwork']['hidden_layer_num'])
        hypernetwork_hidden_layer_size = int(
            config['hypernetwork']['hidden_layer_size'])

        # initialize architect parameters: alphas
        n_ops = len(primitives)
        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()

        self.hyper_normal = []
        self.hyper_reduce = []

        # создаем гиперсети
        for i in range(n_nodes):
            if n_layers >= 3:                
                hypernet = HyperNet(
                    hypernetwork_hidden_layer_num, hypernetwork_hidden_layer_size, i+2, n_ops, self.kernel_num)
                self.alpha_normal.extend(list(hypernet.parameters()))
                self.hyper_normal.append(hypernet)
            hypernet = HyperNet(hypernetwork_hidden_layer_num,
                                hypernetwork_hidden_layer_size, i+2, n_ops, self.kernel_num)
            self.alpha_reduce.extend(list(hypernet.parameters()))
            self.hyper_reduce.append(hypernet)

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:                
                self._alphas.append((n, p))

    def forward(self, x, lam = None):
        if lam is None:
            # проверка: в обучении forward всегда вызывается с заданной лямбдой. 
            # если лямбда не задана, скорее всего производится оценка качества            
            if self.training:
                raise ValueError('Cannot use default lambda value during training')
            lam = torch.zeros((1,1)).to(self.device)
        else:
            lam = self.norm_lam(lam)
            
        if self.sampling_mode == 'softmax':
            weights_normal = [F.softmax(alpha(lam)/self.t, dim=-1)
                              for alpha in self.hyper_normal]
            weights_reduce = [F.softmax(alpha(lam)/self.t, dim=-1)
                              for alpha in self.hyper_reduce]
        elif self.sampling_mode == 'gumbel-softmax':
            weights_normal = [torch.distributions.RelaxedOneHotCategorical(
                self.t, logits=alpha(lam)).rsample([x.shape[0]]) for alpha in self.hyper_normal]
            weights_reduce = [torch.distributions.RelaxedOneHotCategorical(
                self.t, logits=alpha(lam)).rsample([x.shape[0]]) for alpha in self.hyper_reduce]            
        elif self.sampling_mode == 'naive':
            weights_normal = [alpha(lam) for alpha in self.hyper_normal]
            weights_reduce = [alpha(lam) for alpha in self.hyper_reduce]
        else:
            raise ValueError('Bad sampling mode')

        return self.net(x, lam, weights_normal, weights_reduce)

    def loss(self, X, y, lam):
        logits = self.forward(X, lam)
        return self.criterion(logits, y)

    def norm_lam(self, lam):  
        return lam
    
    
    def hyperloss(self, X, y, lam):
        logits = self.forward(X, lam)
        penalty = 0
        latency_loss = 0.0
        for id, cell in enumerate(self.net.cells):
            # можно не пробегать несколько раз, т.к. клетки одинаковы (С точностью до normal и reduce)            
            
            lam_ = self.norm_lam(lam)
            weights = [alpha(lam_) for alpha in self.hyper_reduce] if cell.reduction else [
                alpha(lam_) for alpha in self.hyper_normal]
            
            weights = [F.softmax(w, dim=-1) for w in weights]
            
            
                
            for edges, w_list in zip(cell.dag, weights):
                for mixed_op, weights in zip(edges, w_list):
                    for op, w in zip(mixed_op._ops, weights):
                        for param in op.parameters():
                            penalty += w*np.prod(param.shape)#(torch.norm(param)**2)
                        op_name = op.name if hasattr(op, 'name') else str(op.__class__.__name__).lower()
                        if 'zero' in op_name:
                            op_name = 'none'
                        latency_loss += w * lam[self.connect_dict[op_name]]

        total_task_loss = self.criterion(logits, y)
        #print ('TODO FIX ?')

        latency_penalty = 0.0
        latency_penalty = self.latency_kappa * latency_loss
        total_loss = total_task_loss + latency_penalty
        print("Task loss: {:.4f}, Latency loss: {:.4f}".format(total_task_loss.item(), latency_penalty))
    
        return total_loss
##################################
    def train_step(self, trn_X, trn_y, val_X, val_y, lam=None):
        self.w_optim.zero_grad()
        self.alpha_optim.zero_grad()
    
        loss_total = 0.0
        for _ in range(self.lam_sample_num):
            if lam is None:
                k = torch.distributions.Exponential(1.0).sample([SIMPLEX_D]).to(self.device)
                lam = k/sum(k).to(self.device)
                # print(SIMPLEX_D)
                # print(lam)
                # print(lam.shape)
                # print()
            loss = self.hyperloss(trn_X, trn_y, lam) / self.lam_sample_num
            loss_total += loss
        
        loss_total.backward()
        nn.utils.clip_grad_norm_(list(self.weights()), self.w_grad_clip)
    
        self.w_optim.step()
        self.alpha_optim.step()
    
        return loss_total
#####################################
    def new_epoch(self, e, w, l):
        print(f"[Epoch {e}] Текущая температура (gumbel) = {self.t}")
        SearchCNNController.new_epoch(self, e,w,l)
        self.cur_e = e
        
    def genotype(self, lam, mode='DARTS'):
        w_normal, w_reduce = [], []
        if mode == 'DARTS':
            for w_out, alphas in zip((w_normal, w_reduce), (self.hyper_normal, self.hyper_reduce)):                
                for alpha in alphas:
                    edges = F.softmax(alpha(lam))
                    print (edges)
                    edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
                    topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), 2) # get top-2
                    w_out.append([edges.shape[1]-1]*(len(edges)))
                    for k in topk_edge_indices:                        
                        w_out[-1][k.item()] = primitive_indices[k.item()][0].item()
        elif mode == 'simple':
            for alpha in self.hyper_reduce:
                alpha = alpha(lam)
                w_reduce.append((torch.argmax(alpha, 1).cpu().detach().numpy()).tolist())
            for alpha in self.hyper_normal:
                alpha = alpha(lam)
                w_normal.append((torch.argmax(alpha, 1).cpu().detach().numpy()).tolist())    
        else:
            raise NotImplementedError('Unknown genotype extraction mode:'+mode)
        return w_reduce, w_normal