import os
import numpy as np

# {'max_pool_3x3': 0, 'avg_pool_3x3': 1, 'skip_connect': 2, 'sep_conv_3x3': 3, 'sep_conv_5x5': 4, 'dil_conv_3x3': 5, 'dil_conv_5x5': 6, 'none': 7}
l_vectors = [
	[ 1 , 1 , 0.1 , 3 , 5 , 3 , 5 , 0.1 ],
	[ 5 , 5 , 0.1 , 1 , 1 , 1 , 1 , 0.1],
	[ 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 ],
	[ 1 , 1 , 0.1 , 1 , 1 , 1 , 1 , 0.1 ],
]
for l in l_vectors: # [0.0,  0.25, 0.5, 0.75,  1.0]:
	l = np.array(l)/sum(l)
	cmd = 'python analysis/hyper/make_genotype.py  ./configs/my_configs/my_comp.cfg  ./searchs/my_normal_computer/checkpoint_0_9.ckp simple  {0}  ./searchs/my_normal_computer/genotype_0_{1}.json'.format(l, str(l).replace('.', '_'))
	print ('\n',cmd , '\n')
	os.system(cmd)
	print('\n')

