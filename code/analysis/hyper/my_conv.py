import os
for l in [0.0,  0.25, 0.5, 0.75,  1.0]:
	cmd = 'python3 analysis/hyper/make_genotype.py  ./configs/mini_fmnist_hyper_final/my_diff_conv.cfg  ./searchs/my_high_conv/checkpoint_0_9.ckp simple  {0}  ./searchs/my_high_conv/genotype_0_{1}.json'.format(l, str(l).replace('.', '_'))
	print (cmd)
	os.system(cmd)

