import os
import time, math
import numpy as np
from torch.backends import cudnn

import warnings

from option import opt

from scipy.stats import pearsonr

from models.GMImpute import *


from dataset import Dataset_geno
from torch.utils.data import DataLoader

from scipy.io import savemat


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

warnings.filterwarnings('ignore')
IMGSIZE = {}
IMGSIZE['CellularTax'] = [49, 52]

# IMGSIZE['Single_Surv'] = [42, 44]
# IMGSIZE['Effect_3D'] = [48, 48]
# IMGSIZE['Embryo_Body'] = [47, 48]
# IMGSIZE['Dis_Cell'] = [48, 48]
# IMGSIZE['Zebra_fish'] = [48, 48]
# IMGSIZE['SimulationData'] = [48, 48]
# IMGSIZE['SimulationDataln'] = [48, 48]

def test(net,loader_test, target_size, ori_size):
	net.eval()
	# torch.cuda.empty_cache()
	GT_ALL = []
	INPUT_ALL = []
	FAKE_PRED_ALL = []

	for i, (inputs, masks, targets, _) in enumerate(loader_test):
		GT_ALL.append(targets)
		INPUT_ALL.append(inputs)  # * max_num)
		inputs = inputs.to(opt.device)
		# targets = targets.to(opt.device)
		with torch.no_grad():
			pred = net(inputs)
			FAKE_PRED_ALL.append(pred.cpu())# * max_num)

	INPUT_ALL = torch.cat(INPUT_ALL, dim=0)
	GT_ALL = torch.cat(GT_ALL, dim=0)
	FAKE_PRED_ALL = torch.cat(FAKE_PRED_ALL, dim=0)
	if target_size > ori_size:
		cut_ind = (target_size - ori_size) // 2
		GT = GT_ALL[:, 0, cut_ind:cut_ind + ori_size, cut_ind:cut_ind + ori_size].contiguous().view(-1, ori_size * ori_size)
		F = FAKE_PRED_ALL[:, 0, cut_ind:cut_ind + ori_size, cut_ind:cut_ind + ori_size].contiguous().view(-1, ori_size * ori_size)
		INPUT = INPUT_ALL[:, 0, cut_ind:cut_ind + ori_size, cut_ind:cut_ind + ori_size].contiguous().view(-1, ori_size * ori_size)

		GT2 = GT_ALL[:, 0, cut_ind:cut_ind + ori_size, cut_ind:cut_ind + ori_size].contiguous()
		F2 = FAKE_PRED_ALL[:, 0, cut_ind:cut_ind + ori_size, cut_ind:cut_ind + ori_size].contiguous()
		INPUT2 = INPUT_ALL[:, 0, cut_ind:cut_ind + ori_size, cut_ind:cut_ind + ori_size].contiguous()
	else:
		GT = GT_ALL.view(-1, target_size * target_size)
		F = FAKE_PRED_ALL.view(-1, target_size * target_size)
		INPUT = INPUT_ALL.view(-1, target_size * target_size)

		GT2 = GT_ALL[:, 0]
		F2 = FAKE_PRED_ALL[:, 0]
		INPUT2 = INPUT_ALL[:, 0]

	PEARSON_COE = []
	MSE_before = []
	MSE_after = []
	PEARSON_COE_IN = []
	# P_VALUE = []
	INPUT = torch.where(INPUT == 0.999, torch.tensor(0.), INPUT)
	INPUT2 = torch.where(INPUT2 == 0.999, torch.tensor(0.), INPUT2)

	for t in range(INPUT.shape[-1]):
		ipt = INPUT[:, t]
		gt = GT[:, t]
		f = F[:, t]
		MSE_before.append(((ipt - gt) ** 2).mean().item())
		MSE_after.append(((f - gt) ** 2).mean().item())
		pear_co_f, p_f = pearsonr(f, gt)
		pear_co_in, p_in = pearsonr(ipt, gt)
		PEARSON_COE.append(pear_co_f)
		PEARSON_COE_IN.append(pear_co_in)
		# P_VALUE.append([p_masked, p_f])
	PEARSON_COE = np.array(PEARSON_COE)
	PEARSON_COE_IN = np.array(PEARSON_COE_IN)
	print('Before Imputation ==> Pearson: {:.4f} ± {:.4f}'.format(PEARSON_COE_IN.mean(), PEARSON_COE_IN.std()))
	print('After Imputation ==> Pearson: {:.4f} ± {:.4f}'.format(PEARSON_COE.mean(), PEARSON_COE.std()))
	return F2.numpy(), GT2.numpy(), INPUT2.numpy()




def set_seed_torch(seed=2022):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

	set_seed_torch(666)
	start_time = time.time()

	opt.device = 'cuda:0'
	opt.model_path = './trained_models/' + opt.dataset + '_' + opt.rate + '_GMImpute_pearson_coe_' + opt.epoch_suffix + '.pth'



	opt.data_root = './data/' + opt.dataset + '_' + 'dataSAVER' + opt.rate + '.mat'
	opt.eval_data_root = './data/' + opt.dataset + '_' + 'dataSAVER' + opt.rate + '.mat'

	opt.target_size = IMGSIZE[opt.dataset][1]
	opt.ori_size = IMGSIZE[opt.dataset][0]


	dataloader = DataLoader(
		Dataset_geno(opt.data_root, opt.target_size, opt.ori_size, dataset='genoMap',
					 mask_reverse=True, training=False, creat_mask=False, reverse=True, assi=0.999, eval_train=True, lowerb=0., ratio=opt.Hratio,
					 Lratio=opt.Lratio), batch_size=opt.batch_size,
		shuffle=False, num_workers=opt.n_threads)

	eval_dataloader = DataLoader(
		Dataset_geno(opt.eval_data_root, opt.target_size, opt.ori_size, dataset='genoMap', mask_reverse=True,
					 training=False, creat_mask=False,  reverse=True, assi=0.999, lowerb=0.,
					 ratio=opt.Hratio, Lratio=opt.Lratio,  for_val=True),
		batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_threads)

	if not os.path.exists(opt.model_path):
		print("Invalid checkpoint!")
		exit()

	net = GMImpute(1, 1)
	net.load_state_dict(torch.load(opt.model_path, 'cpu')['model'])


	net = net.to(opt.device)
	epoch_size = len(dataloader)

	# if opt.device == 'cuda':
	# 	net = torch.nn.DataParallel(net)
	# 	cudnn.benchmark = True

	pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print("Total_params: ==> {}".format(pytorch_total_params))

	print("==============================Start evaluation!==============================")
	print("\nDataset: ", opt.dataset, " Efficiency loss: ", float(opt.rate.split('-')[0])/float(opt.rate.split('-')[1]))
	with torch.no_grad():
		print("\n********Evaluation for training dataset!********")
		Pred_train, GT_train, Ori_train = test(net, dataloader, opt.target_size, opt.ori_size)
		print("*******************Finished!********************")
		print("\n*********Evaluation for test dataset!*********")
		Pred_test, GT_test, Ori_test = test(net, eval_dataloader, opt.target_size, opt.ori_size)
		print("*******************Finished!*******************")
		savemat(opt.save_dir + '/' + opt.dataset + '_' + opt.rate + '.mat', {'Pred_train': Pred_train, 'Ori_train': Ori_train, 'GT_train': GT_train, 'Pred_test': Pred_test, 'Ori_test': Ori_test, 'GT_test': GT_test})
	print("\nResults saved!")
	print("\n================================End evaluation!==============================")