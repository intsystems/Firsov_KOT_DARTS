import os
for l in [0.0,  0.25, 0.5, 0.75,  1.0]:
	cmd = 'python analysis/hyper/make_genotype.py  ./configs/mini_fmnist_hyper_final/my_comp.cfg  ./searchs/my_normal_computer/checkpoint_0_9.ckp simple  {0}  ./searchs/my_normal_computer/genotype_0_{1}.json'.format(l, str(l).replace('.', '_'))
	print (cmd)
	os.system(cmd)

