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
	print ('args: <path to config> <path to checkpoint> <mode> <normzlized lambda> <path to save>')
	config = configobj.ConfigObj(sys.argv[1])
	config['device'] = 'cpu'
	model = SearchCNNControllerWithHyperNet(**config)
	model.load_state_dict(torch.load(sys.argv[2], map_location='cpu'))
	red, norm = model.genotype(torch.tensor(float(sys.argv[4])), mode=sys.argv[3])
	print ('param num', calc_param_number(model, red, norm))
	with open(sys.argv[-1], 'w') as out:
		out.write(json.dumps([red,norm]))		
