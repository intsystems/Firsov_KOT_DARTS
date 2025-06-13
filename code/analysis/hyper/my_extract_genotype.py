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
    
if __name__=='__main__':
	# l_vectors = [
	# # [[ 1 , 1 , 3 , 30 , 50 , 30 , 50 , 3 ], "большой штраф за свёртки, ожидается МАЛО параметров (skip/none - 3)"],
	# [[ 1 , 1 , 1 , 3 , 5 , 3 , 5 , 1 ], "чуть меньше штраф за свёртки, ожиается Мало параметров (skip/none - 1)"],
    # # [[ 1 , 1 , 0.1 , 30 , 50 , 30 , 50 , 0.1 ], "большой штраф за свёртки, ожидается МАЛО параметров (skip/none - 0.1)"],
	# # [[ 1 , 1 , 0.1 , 3 , 5 , 3 , 5 , 0.1 ], "чуть меньше штраф за свёртки, ожиается чуть БОЛЬШЕ параметров (skip/none - 0.1)"],
    # [[ 1 , 1 , 0.1 , 1 , 1 , 1 , 1 , 0.1 ], 'за всё одинаково, кроме skip/none (skip/none - 0.1)'],
	# # [[ 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ], 'за всё 1'],
	# [[ 5 , 5 , 0.1 , 1 , 1 , 1 , 1 , 0.1], 'чуть больше штраф за пулинги, ожидается МНОГО параметров (но мало за skip/none)'],
	# # [[ 50 , 50 , 0.1 , 1 , 1 , 1 , 1 , 0.1], "большой штраф за пулинги, ожидается ЕЩЁ БОЛЬШЕ параметров (но мало за skip/none)"],	
	# # [[ 5 , 5 , 1 , 1 , 1 , 1 , 1 , 1], 'чуть больше штраф за пулинги, ожидается МНОГО параметров (skip/none - 1)'],
	# # [[ 50 , 50 , 3 , 1 , 1 , 1 , 1 , 3], "большой штраф за пулинги, ожидается ЕЩЁ БОЛЬШЕ параметров (skip/none - 3)"],
	# # [[ 1 , 100 , 100 , 100 , 100 , 100 , 100 , 100], 'за всё одинаково много кроме 1'],
	# [[ 1 , 1 , 1 , 100000000000 , 100000000000 , 100000000000 , 100000000000 , 1], 'за всё одинаково, много кроме 1 2'],
	# ]
	l_vectors = [
		[[1,1,1], '[1,1,1] одинак'],
		[[1,1,5], '1 1 5 штраф за свёртку'],
		[[1, 1, 100],'1 1 100 штраф за свёртку'],
		[[1, 100, 100],'1 100 100 штраф кроме пулинга'],
		[[5,1,1],'5 1 1 штраф за пулинг'],
		[[100,1,1],'100 1 1 штраф за пулинг'],
		[[100,100,1],'100,100,1 штраф кроме свёртки'],

	]
	l_vectors = [
		[[1,1,1,1,1], 'одинак'],
		[[1,1,1,5,5], ' штраф за свёртку'],
		[[1, 1, 5, 1,1],'штраф за ident'],
		[[5,5,1,1,1],'штраф за пулинг'],
		[[1,1,1,5,1], ' штраф за одну свёртку'],
		[[1, 1, 1, 1,5],'штраф за другую свёртку'],
		[[5,1,1,1,1],'штраф за один пулинг'],
		[[1,5,1,1,1],'штраф за другой пулинг'],


	]
	# {'avg_pool_3x3': 0, 'max_pool_3x3': 1, 'skip_connect': 2, 'sep_conv_3x3': 3, 'sep_conv_5x5': 4}
	# [[ 1 , 1 , 3 , {'avg_pool_3x3': 0, 'skip_connect': 1, 'sep_conv_3x3': 2}
     
	# {'max_pool_3x3': 0, 'avg_pool_3x3': 1, 'skip_connect': 2, 'sep_conv_3x3': 3, 'sep_conv_5x5': 4, 'dil_conv_3x3': 5, 'dil_conv_5x5': 6, 'none': 7}
	for i, elem in enumerate(l_vectors):
		l, text = elem
		lam = np.array(l)/sum(l)
		#print ('args: <path to config> <path to checkpoint> <mode> <normzlized lambda> <path to save>')
            
        # kappa_10
		kappa = '50_3_ops'# 10 # 100, 1000, 10000 , 100000, 5_again
		
		path_to_cfg =  './configs/fmnist_5_ops/kappa_50.cfg'
		path_to_checkpoint =  f'./searchs/5_ops_kappa_50/checkpoint_0_19.ckp'
		path_to_save = f'./searchs/5_ops_kappa_50/genotype'

		config = configobj.ConfigObj(path_to_cfg)
		config['device'] = 'cpu'
		model = SearchCNNControllerWithHyperNet(**config)
		model.load_state_dict(torch.load(path_to_checkpoint, map_location='cpu'))
		lam_tensor = torch.tensor(lam, dtype=torch.float)
		red, norm = model.genotype(lam_tensor, mode='simple')
		rev_dict = {v: k for k, v in model.connect_dict.items()}
        
		print(text)
		print(lam)
		print ('param num', calc_param_number(model, red, norm), '\n')
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

        