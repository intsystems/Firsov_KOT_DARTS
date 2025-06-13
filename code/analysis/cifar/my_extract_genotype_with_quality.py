# extracts genotype from the hypernet. Note, that calc_param_number function is outdated and used only for logging, not for the article results
import torch
import configobj
import sys
sys.path.append('.')
from models.cnn_darts_hypernet.search_cnn_darts_hypernet import SearchCNNControllerWithHyperNet
import json 
import numpy as np 
import torch.nn.functional as F
#####################
import utils
from utils import accuracy, AverageMeter
from torch.utils.data import DataLoader

@torch.no_grad()
def validate(valid_loader, model, lam, device='cpu'):
    """
    Аналогично тому, как это делается в search.py:
    - выключаем train
    - считаем loss и top1 в среднем по mini-batch
    - возвращаем средние значения
    """
    top1 = AverageMeter()
    losses = AverageMeter()
    model.eval()

    for step, (X, y) in enumerate(valid_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X, lam)  # здесь важно, что model(...) требует lam
        loss = model.criterion(logits, y)

        prec1 = accuracy(logits, y)[0]  # top1
        N = X.size(0)
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)

    return top1.avg, losses.avg

def calc_param_number(model, g_reduce, g_normal):
    penalty = 0
    for id, cell in enumerate(model.net.cells):
            weights = g_reduce if cell.reduction else g_normal
            
            for edges, w_list in zip(cell.dag, weights):
                for mixed_op, weight in zip(edges, w_list):
                    op = mixed_op._ops[weight]

                    for param in op.parameters():
                        penalty += np.prod(param.shape)
                        # print(param.shape, op) 
    return penalty    
    
if __name__=='__main__':

	l_vectors = [
		# [[1,1,1], '[1,1,1] одинак'],
		# [[1,1,5], '1 1 5 штраф за свёртку'],
		# [[1, 1, 100],'1 1 100 штраф за свёртку'],
		# [[1, 100, 100],'1 100 100 штраф кроме пулинга'],
		# [[5,1,1],'5 1 1 штраф за пулинг'],
        [[1,3,1],'1 5 1 штраф за ident'],
        [[1,4,1],'1 5 1 штраф за ident'],
        [[1,6,1],'1 5 1 штраф за ident'],
		# [[100,1,1],'100 1 1 штраф за пулинг'],
		# [[100,100,1],'100,100,1 штраф кроме свёртки'],

	]
	path_to_cfg =  f'./configs/my_cifar/cifar_hyper_my.cfg'
	path_to_checkpoint =  f'./searchs/my_cifar_50_epoch/checkpoint_0_38.ckp' # 9  49  38
	path_to_save = f'./searchs/my_cifar_50_epoch/genotype'

	# [[ 1 , 1 , 3 , {'avg_pool_3x3': 0, 'skip_connect': 1, 'sep_conv_3x3': 2}
	config = configobj.ConfigObj(path_to_cfg)
	device = config.get('device', 'cpu')  # если не прописано, пусть будет cpu
	input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
		config['dataset'], 
		'./data', 
		cutout_length=int(config['cutout']), 
		validation=True
	)
	valid_loader = DataLoader(
		valid_data,
		batch_size=int(config['batch_size']),
		shuffle=False,
		pin_memory=True,
	)
	config['device'] = device
	model = SearchCNNControllerWithHyperNet(**config)
	model.load_state_dict(torch.load(path_to_checkpoint, map_location=device))
	model.to(device)
	model.eval()

	rev_dict = {v: k for k, v in model.connect_dict.items()}
	# {'max_pool_3x3': 0, 'avg_pool_3x3': 1, 'skip_connect': 2, 'sep_conv_3x3': 3, 'sep_conv_5x5': 4, 'dil_conv_3x3': 5, 'dil_conv_5x5': 6, 'none': 7}
	for i, elem in enumerate(l_vectors):
		count_operations_normal = {'avg_pool_3x3': 0, 'skip_connect': 0, 'sep_conv_3x3': 0}
		count_operations_reduce = {'avg_pool_3x3': 0, 'skip_connect': 0, 'sep_conv_3x3': 0}
		l, text = elem
		lam = np.array(l)/sum(l)
		#print ('args: <path to config> <path to checkpoint> <mode> <normzlized lambda> <path to save>')
            
        # kappa_10
		
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
			# print(f"  Слой {layer_idx}:")
			for edge_idx, (op_idx, prob) in enumerate(zip(chosen, chosen_probs)):
				op_name = rev_dict.get(op_idx, f"op{op_idx}")
				count_operations_reduce[op_name] += 1
				# print(f"    Ребро {edge_idx}: {op_name}, {prob:.2f}%", end='  |||  ')
			red_ops_summary.append(chosen)
		print(f"  Суммарное количество операций: {count_operations_reduce}\n")
            
		print("\nВыбор операций для hyper_normal:")
		norm_ops_summary = []
		for layer_idx, alpha in enumerate(model.hyper_normal):
			a = alpha(lam_tensor)
			probs = F.softmax(a, dim=1)
			chosen = torch.argmax(probs, dim=1).cpu().detach().numpy().tolist()
			chosen_probs = [float(probs[row, idx]) * 100 for row, idx in enumerate(chosen)]
			# print(f"  Слой {layer_idx}:")
			for edge_idx, (op_idx, prob) in enumerate(zip(chosen, chosen_probs)):
				op_name = rev_dict.get(op_idx, f"op{op_idx}")
				count_operations_normal[op_name] += 1
				# print(f"    Ребро {edge_idx}: {op_name}, {prob:.2f}%", end='  |||  ')
			norm_ops_summary.append(chosen)
		print(f"  Суммарное количество операций: {count_operations_normal}\n")
#################################################################################
		val_top1, val_loss = validate(valid_loader, model, lam_tensor, device=device)
		print(f"Validation accuracy = {val_top1:.4%}, loss = {val_loss:.4f}")
        #"Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

		with open(path_to_save + f'_{i}_' + '.json' , 'w') as out:
			out.write(json.dumps([red,norm]))	

        