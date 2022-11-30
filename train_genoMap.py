import os, warnings
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
from option import opt

from scipy.stats import pearsonr
from models.GMImpute import *

from dataset import Dataset_geno
from torch.utils.data import DataLoader
import json


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

warnings.filterwarnings('ignore')
IMGSIZE = {}
IMGSIZE['CellularTax'] = [49, 52]


# IMGSIZE['Single_Surv'] = [42, 44]
# IMGSIZE['Effect_3D'] = [48, 48]
# IMGSIZE['Embryo_Body'] = [47, 48]
# IMGSIZE['Dis_Cell'] = [48, 48]
# IMGSIZE['Zebra_fish'] = [48, 48]
# IMGSIZE['SimulationData'] = [48, 48]



def train(net, loader_train, loader_test, optim, criterion, target_size, ori_size):
	losses = []
	start_step = 0
	max_pearson_coe, max_pearson_coe_std = -1, -1
	pearsons = []
	print(os.path.exists(opt.model_dir))

	if opt.resume and os.path.exists(opt.model_pretrain):

		ckp = torch.load(opt.model_pretrain)
		print(f'resume from {opt.model_dir}')
		losses = ckp['losses']
		net.load_state_dict(ckp['model'])
		optim.load_state_dict(ckp['optimizer'])
		start_step = ckp['step']
		max_pearson_coe = ckp['max_pearson_mean']
		max_pearson_coe_std = ckp['max_pearson_coe_std ']
		print(f'start_step:{start_step} start training ---')
	else:
		print('train from scratch *** ')

	epoch = 0
	lr = opt.lr
	for step in range(start_step+1, steps+1):
		net.train()
		if epoch == 40:
			lr = opt.lr * 0.5
			for param_group in optim.param_groups:
				param_group["lr"] = lr
		if epoch == 80:
			lr = opt.lr * 0.25
			for param_group in optim.param_groups:
				param_group["lr"] = lr
		x, mask, y = next(iter(loader_train))

		y = y.to(opt.device)
		x = x.to(opt.device)

		out = net(x)

		loss_rec = criterion[0](out, y)

		label_fft = torch.rfft(y, signal_ndim=2, normalized=False, onesided=False)
		pred_fft = torch.rfft(out, signal_ndim=2, normalized=False, onesided=False)

		f_loss = criterion[0](pred_fft, label_fft)

		loss = opt.w_loss_l1*loss_rec + opt.w_loss_f * f_loss

		loss.backward()

		optim.step()
		optim.zero_grad()
		losses.append(loss.item())

		print(
			f'\rloss:{loss.item():.5f} l1:{opt.w_loss_l1 * loss_rec.item():.5f} l1_fft:{opt.w_loss_f * f_loss.item():.5f} | step :{step}/{steps}|lr :{lr :.7f} | time_used :{(time.time() - start_time) / 60 :.1f}',
			end='')

		if step % opt.eval_step == 0:
			epoch = int(step / opt.eval_step)

			with torch.no_grad():
				pearson_mean, pearson_std = test(net, loader_test, target_size, ori_size)

			log = f'step :{step} | epoch: {epoch} | pearson mean:{pearson_mean:.4f} | pearson std:{pearson_std:.4f}'
			print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
			print(log)
			with open(f'./logs_train/{opt.model_name}.txt', 'a') as f:
				log2 = f'{step},{epoch},{loss.item():.5f},{pearson_mean:.4f}'
				f.write(log2 + '\n')

			pearsons.append([pearson_mean, pearson_std])

			if pearson_mean > max_pearson_coe:
				max_pearson_coe = pearson_mean
				max_pearson_coe_std = pearson_std
				save_model_dir = opt.model_dir + '_pearson_coe_best.pth'

				print(
					f'model saved at step :{step}| epoch: {epoch} | max_pearson_mean:{pearson_mean:.4f} | pearson std:{pearson_std:.4f}')

				torch.save({
					'epoch': epoch,
					'step': step,
					'max_pearson_mean': max_pearson_coe,
					'pearson_std': pearson_std,
					'model': net.state_dict(),
					'optimizer': optim.state_dict()
				}, save_model_dir)
			if epoch == 50:
				save_Newest_model_dir = opt.model_dir + '_pearson_coe_' + str(epoch) + '.pth'
				torch.save({
					'epoch': epoch,
					'step': step,
					'pearson_mean': pearson_mean,
					'pearson_std': pearson_std,
					'model': net.state_dict(),
					'optimizer': optim.state_dict()
				}, save_Newest_model_dir)
			save_Newest_model_dir = opt.model_dir + '_pearson_coe_Newest.pth'
			torch.save({
				'epoch': epoch,
				'step': step,
				'pearson_mean': pearson_mean,
				'pearson_std': pearson_std,
				'model': net.state_dict(),
				'optimizer': optim.state_dict()
			}, save_Newest_model_dir)

			print(f'at step :{step}| epoch: {epoch} | Best_pearson_mean:{max_pearson_coe:.4f} | pearson std:{max_pearson_coe_std:.4f}')
			print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

def test(net,loader_test, target_size, ori_size):
	net.eval()
	# torch.cuda.empty_cache()
	GT_ALL = []
	MASKED_ALL = []
	FAKE_PRED_ALL = []

	for i, (inputs, masks, targets, _) in enumerate(loader_test):

		GT_ALL.append(targets)

		MASKED_ALL.append(inputs)
		inputs = inputs.to(opt.device)

		with torch.no_grad():
			pred = net(inputs)
			FAKE_PRED_ALL.append(pred.cpu())

	MASKED_ALL = torch.cat(MASKED_ALL, dim=0)
	GT_ALL = torch.cat(GT_ALL, dim=0)
	FAKE_PRED_ALL = torch.cat(FAKE_PRED_ALL, dim=0)
	if target_size > ori_size:
		cut_ind = (target_size - ori_size) // 2
		GT = GT_ALL[:, 0, cut_ind:cut_ind + ori_size, cut_ind:cut_ind + ori_size].contiguous().view(-1, ori_size * ori_size)
		F = FAKE_PRED_ALL[:, 0, cut_ind:cut_ind + ori_size, cut_ind:cut_ind + ori_size].contiguous().view(-1, ori_size * ori_size)
		MASKED = MASKED_ALL[:, 0, cut_ind:cut_ind + ori_size, cut_ind:cut_ind + ori_size].contiguous().view(-1, ori_size * ori_size)
	else:
		GT = GT_ALL.view(-1, target_size * target_size)
		F = FAKE_PRED_ALL.view(-1, target_size * target_size)
		MASKED = MASKED_ALL.view(-1, target_size * target_size)
	PEARSON_COE = []

	for t in range(MASKED.shape[-1]):
		gt = GT[:, t]
		f_p = F[:, t]
		pear_co_f, p_f = pearsonr(f_p, gt)
		PEARSON_COE.append(pear_co_f)


	PEARSON_COE = np.array(PEARSON_COE)

	return PEARSON_COE.mean(), PEARSON_COE.std()


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
	opt.model_name = opt.dataset + '_' + opt.rate +'_GMImpute'

	model_name = opt.model_name

	log_dir = 'logs_train/' + opt.model_name
	if not os.path.exists(log_dir):
		os.mkdir(log_dir)

	opt.data_root = './data/' + opt.dataset + '_' + 'dataSAVER' + opt.rate + '.mat'
	opt.eval_data_root = './data/' + opt.dataset + '_' + 'dataSAVER' + opt.rate + '.mat'

	opt.target_size = IMGSIZE[opt.dataset][1]
	opt.ori_size = IMGSIZE[opt.dataset][0]
	opt.ema_decay = 0.99
	opt.model_dir = './trained_models/' + opt.model_name

	if not os.path.exists('./logs_train'):
		os.mkdir('./logs_train')

	print(opt)
	print('model_dir:', opt.model_dir)
	print(f'log_dir: {log_dir}')

	if not opt.resume and os.path.exists(f'./logs_train/{opt.model_name}.txt'):
		print(f'./logs_train/{opt.model_name}.txt 已存在，请删除该文件……')
		# exit()

	with open(f'./logs_train/args_{opt.model_name}.txt', 'w') as f:
		json.dump(opt.__dict__, f, indent=2)

	dataloader = DataLoader(Dataset_geno(opt.data_root, opt.target_size, opt.ori_size,  dataset='genoMap', mask_reverse=True, creat_mask=True,  reverse=True, assi=0.999, lowerb=0., ratio=opt.Hratio, Lratio=opt.Lratio), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_threads)

	eval_dataloader = DataLoader(Dataset_geno(opt.eval_data_root, opt.target_size, opt.ori_size, dataset='genoMap', mask_reverse=True,  training=False, creat_mask=False, reverse=True, assi=0.999, lowerb=0., ratio=opt.Hratio, Lratio=opt.Lratio), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_threads)


	net = GMImpute(1, 1)
	# net.load_state_dict(
	# 	torch.load('/home/wei/GMImpute/trained_models/CellularTax_GMImpute_10-2000_epoch50.pth', 'cpu')['model'])

	net = net.to(opt.device)

	epoch_size = len(dataloader)
	opt.eval_step = 50

	steps = opt.eval_step * opt.epochs
	T = opt.eval_step * (opt.epochs * 3)
	print("epoch_size: ", epoch_size)
	# if opt.device == 'cuda':
	# 	net = torch.nn.DataParallel(net)
	# 	cudnn.benchmark = True

	pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print("Total_params: ==> {}".format(pytorch_total_params))

	criterion = []
	criterion.append(nn.L1Loss().to(opt.device))


	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)

	optimizer.zero_grad()
	train(net, dataloader, eval_dataloader, optimizer, criterion, opt.target_size, opt.ori_size)
	

