# extracts genotype from the hypernet. Note, that calc_param_number function is outdated and used only for logging, not for the article results
import torch
import configobj
import sys
sys.path.append('.')
from models.cnn_darts_hypernet.search_cnn_darts_hypernet import SearchCNNControllerWithHyperNet
import json 
import numpy as np 
import torch.nn.functional as F

def calc_param_number(model, g_reduce, g_normal):
    penalty = 0
    for id, cell in enumerate(model.net.cells):
            weights = g_reduce if cell.reduction else g_normal
            
            for edges, w_list in zip(cell.dag, weights):
                for mixed_op, weight in zip(edges, w_list):
                    op = mixed_op._ops[weight]

                    for param in op.parameters():
                        penalty += np.prod(param.shape) 
    return penalty    

def calc_param_number_full(model, g_reduce, g_normal):
    penalty = 0
    stemm_param = 0
    cls_param = 0
    for cell in model.net.cells:
        weights = g_reduce if cell.reduction else g_normal
        for edges, op_indices in zip(cell.dag, weights):
            for mixed_op, op_index in zip(edges, op_indices):
                op = mixed_op._ops[op_index]
                for param in op.parameters():
                    penalty += np.prod(param.shape)
                    
    for param in model.net.stem.parameters():
        stemm_param += np.prod(param.shape)
    for param in model.net.linear.parameters():
        cls_param += np.prod(param.shape)
    
    return stemm_param, penalty, cls_param
if __name__=='__main__':
    l_vectors_5 = [  # {'avg_pool_3x3': 0, 'max_pool_3x3': 1, 'skip_connect': 2, 'sep_conv_3x3': 3, 'sep_conv_5x5': 4}
        [[ 1 , 1 , 100 , 300  , 300  ], "штраф за свёртки, ожиается Мало параметров"],
        [[ 1 , 1 , 100 , 1 , 1  ], 'за всё 1'],
        [[ 500 , 500 , 100 , 1 , 1 ], 'штраф за пулинги, ожидается МНОГО параметров'],
    ]
        # {'max_pool_3x3': 0, 'avg_pool_3x3': 1, 'skip_connect': 2, 'sep_conv_3x3': 3, 'sep_conv_5x5': 4, 'dil_conv_3x3': 5, 'dil_conv_5x5': 6, 'none': 7}
    l_vectors_7 = [
    [[ 1 , 1 , 1 , 300 , 500 , 300 , 500 , 1 ], "штраф за свёртки, ожиается Мало параметров"],
    [[ 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ], 'за всё 1'],
    [[ 500 , 500 , 1 , 1 , 1 , 1 , 1 , 1], 'штраф за пулинги, ожидается МНОГО параметров'],
    ]
    l_vectors_3 = [
        [[1,1,1], '[1,1,1] одинак'],
        [[1,1,5], '1 1 5 штраф за свёртку'],
        # [[1, 1, 100],'1 1 100 штраф за свёртку'],
        # [[1, 100, 100],'1 100 100 штраф кроме пулинга'],
        [[5,1,1],'5 1 1 штраф за пулинг'],
        # [[100,1,1],'100 1 1 штраф за пулинг'],
        # [[100,100,1],'100,100,1 штраф кроме свёртки'],

    ]
    # [[ 1 , 1 , 3 , {'avg_pool_3x3': 0, 'skip_connect': 1, 'sep_conv_3x3': 2}
    S_dim = 7
    if S_dim == 7:
        l_vectors = l_vectors_7
    elif S_dim == 5:
        l_vectors = l_vectors_5
    elif S_dim == 3:
        l_vectors = l_vectors_3

    for i, elem in enumerate(l_vectors):
        l, text = elem
        lam = np.array(l)/sum(l)
        #print ('args: <path to config> <path to checkpoint> <mode> <normzlized lambda> <path to save>')
            
        # kappa_10
        # kappa = '50_3_ops'# 10 # 100, 1000, 10000 , 100000, 5_again

        
        path_to_cfg =  f'configs/cifar_hyper_final/cifar_hyper_kappa.cfg'
        path_to_checkpoint =  f'./searchs/hyper_kappa/checkpoint_0_10.ckp' # 9  49  38
        path_to_save = f'./searchs/hyper_kappa/genotype'
        
        path_to_cfg =  f'configs/fmnist_5_ops/fmnist_kappa_50.cfg'
        path_to_checkpoint =  f'./searchs/fmnist_final/checkpoint_0_19.ckp' # 9  49  38
        path_to_save = f'./searchs/fmnist_final/genotype'
        
        path_to_cfg =  f'configs/cifar_hyper_final/cifar_3_ops_50_kappa.cfg'
        path_to_checkpoint =  f'./searchs/cifar_3_ops_50_kappa/checkpoint_0_29.ckp' # 9  49  38
        path_to_save = f'./searchs/cifar_3_ops_50_kappa/genotype'

        # path_to_cfg =  f'configs/cifar_hyper_final/cifar_3_ops_250_kappa.cfg'
        # path_to_checkpoint =  f'./searchs/cifar_3_ops_250_kappa/checkpoint_0_49.ckp' # 9  49  38
        # path_to_save = f'./searchs/cifar_3_ops_250_kappa/genotype'

        path_to_cfg =  f'configs/cifar_hyper_final/cifar_hyper_kappa_250.cfg'
        path_to_checkpoint =  f'./searchs/cifar_kappa_250/checkpoint_0_49.ckp' # 9  49  38
        path_to_save = f'./searchs/cifar_kappa_250/genotype'

        path_to_cfg =  f'configs/cifar_hyper_final/cifar_hyper_kappa_1000.cfg'
        path_to_checkpoint =  f'./searchs/cifar_kappa_1000/checkpoint_0_49.ckp' # 9  49  38
        path_to_save = f'./searchs/cifar_kappa_1000/genotype'
        
        path_to_cfg =  f'configs/cifar_hyper_final/cifar_3_ops_50_kappa_t_05.cfg'
        path_to_checkpoint =  f'./searchs/cifar_3_ops_50_kappa_t_05/checkpoint_0_49.ckp' # 9  49  38
        path_to_save = f'./searchs/cifar_3_ops_50_kappa_t_05/genotype'

        path_to_cfg =  f'configs/cifar_hyper_final/cifar_3_ops_250_kappa_t_05.cfg'
        path_to_checkpoint =  f'./searchs/cifar_3_ops_250_kappa_t_05/checkpoint_0_49.ckp' # 9  49  38
        path_to_save = f'./searchs/cifar_3_ops_250_kappa_t_05/genotype'

        path_to_cfg =  f'configs/cifar_hyper_final/cifar_3_ops_50_kappa.cfg'
        path_to_checkpoint =  f'./searchs/cifar_3_ops_50_kappa/checkpoint_0_49.ckp' # 9  49  38
        path_to_save = f'./searchs/cifar_3_ops_50_kappa/genotype'

        path_to_cfg =  f'configs/cifar_hyper_final/cifar_3_ops_250_kappa.cfg'
        path_to_checkpoint =  f'./searchs/cifar_3_ops_250_kappa/checkpoint_0_49.ckp' # 9  49  38
        path_to_save = f'./searchs/cifar_3_ops_250_kappa/genotype'

        path_to_cfg =  f'configs/cifar_hyper_final/cifar_3_ops_50_kappa_20_epoch.cfg'
        path_to_checkpoint =  f'./searchs/cifar_3_ops_50_kappa_20_epoch/checkpoint_0_19.ckp' # 9  49  38
        path_to_save = f'./searchs/cifar_3_ops_50_kappa_20_epoch/genotype'

        path_to_cfg =  f'configs/fmnist_5_ops/fmnist_3_ops_kappa_50.cfg'
        path_to_checkpoint =  f'./searchs/fmnist_3_ops_kappa_50/checkpoint_0_19.ckp' # 9  49  38
        path_to_save = f'./searchs/fmnist_3_ops_kappa_50/genotype'
        

        path_to_cfg =  f'configs/fmnist_5_ops/fmnist_3_ops_kappa_50_like_local.cfg'
        path_to_checkpoint =  f'./searchs/fmnist_3_ops_kappa_50_like_local/checkpoint_0_9.ckp' # 9  49  38
        path_to_save = f'./searchs/fmnist_3_ops_kappa_50_like_local/genotype'

        #############################
        name = 'cifar_hyper_all_ops_kappa_5'
        path_to_cfg =  f'configs/cifar_hyper_final/{name}.cfg'
        path_to_checkpoint =  f'./searchs/{name}/checkpoint_0_49.ckp' # 9  49  38
        path_to_save = f'./searchs/{name}/genotype_check'
        
        print(f'Exp: {name}')
        # path_to_cfg =  f'configs/fmnist_final/fmnist_all_ops_kappa_50.cfg'
        # path_to_checkpoint =  f'./searchs/fmnist_all_ops_kappa_50/checkpoint_0_49.ckp' # 9  49  38
        # path_to_save = f'./searchs/fmnist_all_ops_kappa_50/genotype'
        
        config = configobj.ConfigObj(path_to_cfg)
        config['device'] = 'cpu'
        model = SearchCNNControllerWithHyperNet(**config)
        model.load_state_dict(torch.load(path_to_checkpoint, map_location='cpu'))
        lam_tensor = torch.tensor(lam, dtype=torch.float)
        red, norm = model.genotype(lam_tensor, mode='simple')
        rev_dict = {v: k for k, v in model.connect_dict.items()}
        
        print(text)
        print(lam)
        print ('param num', calc_param_number_full(model, red, norm), '\n')
        print("\nВыбор операций для hyper_reduce:")
        red_ops_summary = []
        for layer_idx, alpha in enumerate(model.hyper_reduce):
            a = alpha(lam_tensor)
            probs = F.softmax(a, dim=1)
            chosen = torch.argmax(probs, dim=1).cpu().detach().numpy().tolist()
            # Для каждого ребра получаем вероятность выбранной операции
            chosen_probs = [float(probs[row, idx]) * 100 for row, idx in enumerate(chosen)]
            print(f"  Слой {layer_idx}:")
            for edge_idx, (op_idx, prob) in enumerate(zip(chosen, chosen_probs)):
                op_name = rev_dict.get(op_idx, f"op{op_idx}")
                print(f"    Ребро {edge_idx}: {op_name}, {prob:.2f}%")
            red_ops_summary.append(chosen)

        print("\nВыбор операций для hyper_normal:")
        norm_ops_summary = []
        for layer_idx, alpha in enumerate(model.hyper_normal):
            a = alpha(lam_tensor)
            probs = F.softmax(a, dim=1)
            chosen = torch.argmax(probs, dim=1).cpu().detach().numpy().tolist()
            chosen_probs = [float(probs[row, idx]) * 100 for row, idx in enumerate(chosen)]
            print(f"  Слой {layer_idx}:")
            for edge_idx, (op_idx, prob) in enumerate(zip(chosen, chosen_probs)):
                op_name = rev_dict.get(op_idx, f"op{op_idx}")
                print(f"    Ребро {edge_idx}: {op_name}, {prob:.2f}%")
            norm_ops_summary.append(chosen)
#################################################################################
        

        with open(path_to_save + f'_{i}_' + '.json' , 'w') as out:
            out.write(json.dumps([red,norm]))    

        