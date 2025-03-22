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
    for id, cell in enumerate(model.net.cells):
            weights = g_reduce if cell.reduction else g_normal
            
            for edges, w_list in zip(cell.dag, weights):
                for mixed_op, weight in zip(edges, w_list):
                    op = mixed_op._ops[weight]

                    for param in op.parameters():
                        penalty += np.prod(param.shape) 
    return penalty    
    
if __name__=='__main__':
	l_vectors = [
	[ 1 , 1 , 0.1 , 30 , 50 , 30 , 50 , 0.1 ],
	[ 50 , 50 , 0.1 , 1 , 1 , 1 , 1 , 0.1],
    [ 1 , 1 , 0.1 , 3 , 5 , 3 , 5 , 0.1 ],
	[ 5 , 5 , 0.1 , 1 , 1 , 1 , 1 , 0.1],
	[ 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ],
	[ 1 , 1 , 0.1 , 1 , 1 , 1 , 1 , 0.1 ],
	]
      # {'max_pool_3x3': 0, 'avg_pool_3x3': 1, 'skip_connect': 2, 'sep_conv_3x3': 3, 'sep_conv_5x5': 4, 'dil_conv_3x3': 5, 'dil_conv_5x5': 6, 'none': 7}
	# {'max_pool_3x3': 0, 'avg_pool_3x3': 1, 'skip_connect': 2, 'sep_conv_3x3': 3, 'sep_conv_5x5': 4, 'dil_conv_3x3': 5, 'dil_conv_5x5': 6, 'none': 7}
	for i, l in enumerate(l_vectors):
		lam = np.array(l)/sum(l)
		#print ('args: <path to config> <path to checkpoint> <mode> <normzlized lambda> <path to save>')
            
		path_to_cfg =  './configs/my_configs/big_kappa.cfg'  #'./configs/my_configs/big_kappa.cfg'#'./configs/my_configs/my_comp.cfg'  
		path_to_checkpoint =  './searchs/my_high_conv/checkpoint_0_9.ckp' # './searchs/my_normal_computer/checkpoint_0_9.ckp'
		path_to_save = './searchs/my_high_conv/genotype' # './searchs/my_normal_computer/genotype'
		config = configobj.ConfigObj(path_to_cfg)
		config['device'] = 'cpu'
		model = SearchCNNControllerWithHyperNet(**config)
		model.load_state_dict(torch.load(path_to_checkpoint, map_location='cpu'))
		lam_tensor = torch.tensor(lam, dtype=torch.float)
		red, norm = model.genotype(lam_tensor, mode='simple')
		print(lam)
		print ('param num', calc_param_number(model, red, norm))
		with open(path_to_save + f'_{i}_' + '.json' , 'w') as out:
			out.write(json.dumps([red,norm]))		