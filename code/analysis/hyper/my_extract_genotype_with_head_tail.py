# extracts genotype from the hypernet. Note, that calc_param_number function is outdated and used only for logging, not for the article results
import torch
import configobj
import sys
sys.path.append('.')
from models.cnn_darts_hypernet.search_cnn_darts_hypernet import SearchCNNControllerWithHyperNet
import json 
import numpy as np 
def calc_param_number(model, g_reduce, g_normal):
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
	l_vectors = [
	[[ 1 , 1 , 3 , 30 , 50 , 30 , 50 , 3 ], "большой штраф за свёртки, ожидается МАЛО параметров (skip/none - 3)"],
	[[ 1 , 1 , 1 , 3 , 5 , 3 , 5 , 1 ], "чуть меньше штраф за свёртки, ожиается чуть БОЛЬШЕ параметров (skip/none - 1)"],
    [[ 1 , 1 , 0.1 , 30 , 50 , 30 , 50 , 0.1 ], "большой штраф за свёртки, ожидается МАЛО параметров (skip/none - 0.1)"],
	[[ 1 , 1 , 0.1 , 3 , 5 , 3 , 5 , 0.1 ], "чуть меньше штраф за свёртки, ожиается чуть БОЛЬШЕ параметров (skip/none - 0.1)"],
    [[ 1 , 1 , 0.1 , 1 , 1 , 1 , 1 , 0.1 ], 'за всё одинаково, кроме skip/none (skip/none - 0.1)'],
	[[ 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ], 'за всё 1'],
	[[ 5 , 5 , 0.1 , 1 , 1 , 1 , 1 , 0.1], 'чуть больше штраф за пулинги, ожидается МНОГО параметров (но мало за skip/none)'],
	[[ 50 , 50 , 0.1 , 1 , 1 , 1 , 1 , 0.1], "большой штраф за пулинги, ожидается ЕЩЁ БОЛЬШЕ параметров (но мало за skip/none)"],	
	[[ 5 , 5 , 1 , 1 , 1 , 1 , 1 , 1], 'чуть больше штраф за пулинги, ожидается МНОГО параметров (skip/none - 1)'],
	[[ 50 , 50 , 3 , 1 , 1 , 1 , 1 , 3], "большой штраф за пулинги, ожидается ЕЩЁ БОЛЬШЕ параметров (skip/none - 3)"],
	]
	# {'max_pool_3x3': 0, 'avg_pool_3x3': 1, 'skip_connect': 2, 'sep_conv_3x3': 3, 'sep_conv_5x5': 4, 'dil_conv_3x3': 5, 'dil_conv_5x5': 6, 'none': 7}
	for i, elem in enumerate(l_vectors):
		l, text = elem
		lam = np.array(l)/sum(l)
		#print ('args: <path to config> <path to checkpoint> <mode> <normzlized lambda> <path to save>')
            
		# kappa_5 --- проблемы
		path_to_cfg =  './configs/my_configs/kappa_5.cfg'
		path_to_checkpoint =  './searchs/kappa_5/checkpoint_0_9.ckp'
		path_to_save = './searchs/kappa_5/genotype' 
        # kappa_5_again
		# path_to_cfg =  './configs/my_configs/kappa_5.cfg'
		# path_to_checkpoint =  './searchs/kappa_5_again/checkpoint_0_9.ckp'
		# path_to_save = './searchs/kappa_5_again/genotype'
            
		# kappa_3 что-то с пулингами
		# path_to_cfg =  './configs/my_configs/kappa_3.cfg'
		# path_to_checkpoint =  './searchs/kappa_3/checkpoint_0_9.ckp'
		# path_to_save = './searchs/kappa_3/genotype'
		# #  идеально интерпретируется
		# path_to_cfg =  './configs/my_configs/big_kappa.cfg'
		# path_to_checkpoint =  './searchs/big_kappa/checkpoint_0_9.ckp'
		# path_to_save = './searchs/big_kappa/genotype' 
		# # big_kappa_2   идеально интерпретируется
		# path_to_cfg =  './configs/my_configs/big_kappa.cfg'
		# path_to_checkpoint =  './searchs/big_kappa_2/checkpoint_0_9.ckp'
		# path_to_save = './searchs/big_kappa_2/genotype' 
            
        # kappa_one --- что-то не так!
		# path_to_cfg =  './configs/my_configs/kappa_one.cfg'
		# path_to_checkpoint =  './searchs/kappa_one/checkpoint_0_9.ckp'
		# path_to_save = './searchs/kappa_one/genotype' 
            
		# странное
		# path_to_cfg = './configs/my_configs/big_kappa.cfg'   
		# path_to_checkpoint =  './searchs/my_high_conv/checkpoint_0_9.ckp' 
		# path_to_save = './searchs/my_high_conv/genotype' 
		config = configobj.ConfigObj(path_to_cfg)
		config['device'] = 'cpu'
		model = SearchCNNControllerWithHyperNet(**config)
		model.load_state_dict(torch.load(path_to_checkpoint, map_location='cpu'))
		lam_tensor = torch.tensor(lam, dtype=torch.float)
		red, norm = model.genotype(lam_tensor, mode='simple')
		print(text)
		print(lam)
		print ('stemm_param, penalty, cls_param num', calc_param_number(model, red, norm), '\n')
		with open(path_to_save + f'_{i}_' + '.json' , 'w') as out:
			out.write(json.dumps([red,norm]))		