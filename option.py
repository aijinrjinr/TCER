import os, argparse
import torch, warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str,default='Automatic detection')
parser.add_argument('--resume', type=bool,default=False)
parser.add_argument('--epochs', type=int,default=200)
parser.add_argument('--eval_step', type=int,default=5000)
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--model_dir', type=str,default='./trained_models/')
parser.add_argument('--trainset', type=str,default='its_train')
parser.add_argument('--testset', type=str,default='its_test')
parser.add_argument('--net', type=str,default='GMImpute')
parser.add_argument('--epoch_suffix', type=str,default='Newest')

parser.add_argument('--batch_size', type=int,default=64,help='batch size')
parser.add_argument('--no_lr_sche', action='store_true',help='no lr cos schedule')

parser.add_argument('--model_name', type=str,default='test')
parser.add_argument('--dataset', default='CellularTax')
parser.add_argument('--rate', default='8-2000')
parser.add_argument('--reverse', type=bool, default=True)
parser.add_argument('--model_pretrain', type=str,default='')

parser.add_argument('--w_loss_l1', type=float, default=1)
parser.add_argument('--w_loss_f', type=float, default=0.02)
parser.add_argument('--Hratio', type=float, default=0.25)
parser.add_argument('--Lratio', type=float, default=0.1)


parser.add_argument('--w1', type=float, default=0)
parser.add_argument('--w2', type=float, default=0)
parser.add_argument('--n_threads', type=int, default=0)

parser.add_argument('--pre_train_epochs', type=int, default=10, help='train with l1 and fft')

parser.add_argument('--lr_decay', type=bool, default=True)
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='lr decay rate')
parser.add_argument('--lr_decay_win', type=int, default=4, help='lr decay windows: epoch')

parser.add_argument('--eval_dataset', type=bool, default=False)
parser.add_argument('--GenoMap', type=int, default=1)

opt = parser.parse_args()
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

opt.model_name = 'GMImpute'

opt.model_dir = opt.model_dir + opt.model_name + '.pth'
opt.save_dir = './results/'
log_dir = 'logs_train/'+opt.model_name




if not os.path.exists('trained_models'):
	os.mkdir('trained_models')

if not os.path.exists('logs_train'):
	os.mkdir('logs_train')

if not os.path.exists('results'):
	os.mkdir('results')
