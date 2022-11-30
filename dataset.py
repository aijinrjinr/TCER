import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
# from scipy.misc import imread
from imageio import imread
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt


class Dataset_geno(torch.utils.data.Dataset):
    def __init__(self, image_path, target_size, ori_size,  dataset='', augment=True, training=True,
                 mask_reverse=False, creat_mask=False, reverse=False, assi=0, eval_train=False, scale=False, lowerb=0, ratio=0.2, Lratio=0, for_val=False):
        super(Dataset_geno, self).__init__()
        self.augment = augment
        self.training = training
        self.image_path = image_path
        self.target_size = target_size
        self.ori_size = ori_size
        self.cut_ind = (self.target_size - self.ori_size) // 2
        # self.use_ln = use_ln
        # self.valuemask = valuemask
        # self.multi_scale = multi_scale
        # self.pair = pair
        self.reverse = reverse
        self.assi = assi
        self.eval_train = eval_train
        self.scale = scale
        self.lowerb = lowerb
        self.ratio = ratio
        self.Lratio = Lratio
        # self.joint = joint
        # self.addnoise = addnoise
        # self.ema = ema
        self.for_val = for_val
        # self.dataall = loadmat(image_path)

        if self.training:
            if loadmat(image_path)['test_genoMaps'].shape[-1] == 0:
            # if 'test_genoMaps' not in loadmat(image_path).keys():
                print('!!!!!!!!!!!!!!!!!!!!!!!!!')
                self.dataall = loadmat(image_path)['train_genoMaps']
                self.gtdataall = loadmat(image_path)['train_genoMaps_GT']
            else:
                self.dataall = np.concatenate([loadmat(image_path)['train_genoMaps'], loadmat(image_path)['test_genoMaps']], axis=-1)
                self.gtdataall = np.concatenate([loadmat(image_path)['train_genoMaps_GT'], loadmat(image_path)['test_genoMaps_GT']], axis=-1)
                # self.ratio_range = ratio_range
        else:
            if self.eval_train:
                self.dataall = loadmat(image_path)['train_genoMaps']
                self.gtdataall = loadmat(image_path)['train_genoMaps_GT']
            else:
                # self.dataall = loadmat(image_path)['train_genoMaps']
                # self.gtdataall = loadmat(image_path)['train_genoMaps_GT']
                if self.for_val:
                    self.dataall = loadmat(image_path)['test_genoMaps']
                    self.gtdataall = loadmat(image_path)['test_genoMaps_GT']
                else:
                    # self.dataall = loadmat(image_path)['test_genoMaps']
                    # self.gtdataall = loadmat(image_path)['test_genoMaps_GT']
                    # if 'test_genoMaps' not in loadmat(image_path).keys():
                    if loadmat(image_path)['test_genoMaps'].shape[-1] == 0:
                        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
                        self.dataall = loadmat(image_path)['train_genoMaps']
                        self.gtdataall = loadmat(image_path)['train_genoMaps_GT']
                    else:
                        self.dataall = np.concatenate([loadmat(image_path)['train_genoMaps'], loadmat(image_path)['test_genoMaps']], axis=-1)
                        self.gtdataall = np.concatenate([loadmat(image_path)['train_genoMaps_GT'], loadmat(image_path)['test_genoMaps_GT']], axis=-1)

            self.maskall = (self.dataall > self.lowerb) * 1.0

            # if self.use_ln:
            #     self.dataall = np.log(self.dataall * 10 + 1)
            #     # if not self.training:
            #     self.gtdataall = np.log(self.gtdataall * 10 + 1)
            #     self.maxall = self.dataall.max(0).max(0).max(0)
            # else:
            if not self.training:
                self.maxall = self.dataall.max(0).max(0).max(0)
                # self.dataall = self.dataall / self.dataall.max(0).max(0).max(0)

        self.dataset = dataset

        self.mask_reverse = mask_reverse

        self.creat_mask = creat_mask


    def __len__(self):
        return self.dataall.shape[-1]


    def __getitem__(self, index):
        # try:

        if not self.creat_mask:
            if self.reverse:
                item = self.load_item_reverse(index)
            else:
                item = self.load_item(index)
        else:
            if self.reverse:
                item = self.load_create_item_valuemask_reverse(index)
            else:
                item = self.load_create_item_valuemask(index)


        # except:
        #     print('loading error: ' + self.data[index])
        #     item = self.load_item(0)

        return item


    def padding_image(self, img, gt, mask):
        if self.ori_size > self.target_size:
            img = img[:self.target_size, :self.target_size, ...]
            gt = gt[:self.target_size, :self.target_size, ...]
            mask = mask[:self.target_size, :self.target_size, ...]
        else:
            imgnew = np.zeros([self.target_size, self.target_size, 1])
            gtnew = np.zeros([self.target_size, self.target_size, 1])
            masknew = np.ones([self.target_size, self.target_size, 1])
            imgnew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = img
            gtnew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = gt
            masknew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = mask
            img, gt, mask = imgnew, gtnew, masknew
        return img, gt, mask

    def load_item(self, index):
        gt = self.gtdataall[..., index]
        img = self.dataall[..., index]
        img = np.where(img <= self.lowerb, 0, img)
        if self.scale:
            imgmax = img.max()
            img = img / imgmax
        mask = self.maskall[..., index]
        maxnum = 1#self.maxall[index]
        if self.ori_size > self.target_size:
            img = img[:self.target_size, :self.target_size, ...]
            gt = gt[:self.target_size, :self.target_size, ...]
            mask = mask[:self.target_size, :self.target_size, ...]
        else:
            imgnew = np.zeros([self.target_size, self.target_size, 1])
            gtnew = np.zeros([self.target_size, self.target_size, 1])
            masknew = np.ones([self.target_size, self.target_size, 1])
            imgnew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = img
            gtnew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = gt
            masknew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = mask
            img, gt, mask = imgnew, gtnew, masknew
        # augment data
        # if self.training:
        #     if self.augment and np.random.binomial(1, 0.5) > 0:
        #         img = img[:, ::-1, ...]
        #         gt = gt[:, ::-1, ...]
        #         mask = mask[:, ::-1, ...]
        #     if self.augment and np.random.binomial(1, 0.5) > 0:
        #         img = img[::-1, ...]
        #         gt = gt[::-1, ...]
        #         mask = mask[::-1, ...]
        #
        # img = img / img.max()
        return torch.tensor(img.copy()).permute(2, 0, 1).float(),  torch.tensor(mask.copy()).permute(2, 0, 1).float(), torch.tensor(gt.copy()).permute(2, 0, 1).float(), torch.tensor(maxnum)





    def load_item_reverse(self, index):
        gt = self.gtdataall[..., index]
        img = self.dataall[..., index]
        imgmax = img.max()
        if self.scale:
            img = img / imgmax
        # assi = 0.5
        img = np.where(img <= self.lowerb, self.assi, img)
        mask = self.maskall[..., index]
        maxnum = 1#self.maxall[index]
        if self.ori_size > self.target_size:
            img = img[:self.target_size, :self.target_size, ...]
            gt = gt[:self.target_size, :self.target_size, ...]
            mask = mask[:self.target_size, :self.target_size, ...]
        else:
            imgnew = np.zeros([self.target_size, self.target_size, 1]) + self.assi
            gtnew = np.zeros([self.target_size, self.target_size, 1])
            masknew = np.ones([self.target_size, self.target_size, 1])
            imgnew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = img
            gtnew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = gt
            masknew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = mask
            img, gt, mask = imgnew, gtnew, masknew
        # augment data
        # if self.training:
        #     if self.augment and np.random.binomial(1, 0.5) > 0:
        #         img = img[:, ::-1, ...]
        #         gt = gt[:, ::-1, ...]
        #         mask = mask[:, ::-1, ...]
        #     if self.augment and np.random.binomial(1, 0.5) > 0:
        #         img = img[::-1, ...]
        #         gt = gt[::-1, ...]
        #         mask = mask[::-1, ...]
        #
        # img = img / img.max()

        return torch.tensor(img.copy()).permute(2, 0, 1).float(),  torch.tensor(mask.copy()).permute(2, 0, 1).float(), torch.tensor(gt.copy()).permute(2, 0, 1).float(), imgmax


    def load_create_item_valuemask(self, index):
        gt_real = self.gtdataall[..., index]
        gt = self.dataall[..., index]
        # gt = np.where(gt <= self.lowerb, 0, gt)
        if self.scale:
            gtmax = gt.max()
            gt = gt / gt.max()
        # findex = np.random.uniform(0, 1)
        # if findex < 0.3:
        #     gt = gt[::-1].copy()
        # elif findex < 0.6:
        #     gt = gt[:, ::-1].copy()
        ratio2 = 1#np.random.uniform(1, 1.5, 1)[0]# np.random.uniform(1, 4, 1)[0] # np.random.uniform(1, 4, 1)[0]#1#np.random.uniform(4, 5, 1)[0]
        gt = (gt * ratio2).clip(0, 1)
        # gt = (gt * ratio2).clip(0, 1)
        # scale = random.sample(np.arange(1, gt.max() * 0.6, 0.1).tolist(), 1)[0]
        # idx = np.where(gt > 0)

        idx1 = np.where((gt > self.lowerb))# & (gt < (0.9  * gt.max())))
        # idx1 = np.where((gt >= gt.mean()))
        # idx2 = np.where((gt < gt.mean()))

        img = np.copy(gt)

        # ratio = np.random.uniform(0.0, 0.2, 1)[0]
        ratio1 = np.random.uniform(self.Lratio, self.ratio, 1)[0]#np.random.uniform(0.0, 0.7, 1)[0]
        # ratio2 = np.random.uniform(0.5, 1., 1)[0]
        pick2set0 = random.sample([i for i in range(len(idx1[0]))], int(len(idx1[0]) * ratio1))
        pick2set0.sort()
        indexnew = (idx1[0][pick2set0], idx1[1][pick2set0], idx1[2][pick2set0])
        img[indexnew] = 0
        if self.addnoise:
            addindex = np.where(img > 0)
            img[addindex] = np.random.normal(img.mean(), 0.05, len(addindex[0]))
            img = img.clip(0, 1)

        mask = np.where(gt > self.lowerb, 1, 0)

        if self.ori_size > self.target_size:
            img = img[:self.target_size, :self.target_size, ...]
            gt = gt[:self.target_size, :self.target_size, ...]
            mask = mask[:self.target_size, :self.target_size, ...]
        else:
            imgnew = np.zeros([self.target_size, self.target_size, 1])
            gtnew = np.zeros([self.target_size, self.target_size, 1])
            masknew = np.ones([self.target_size, self.target_size, 1])
            imgnew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = img
            gtnew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = gt
            masknew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = mask
            img, gt, mask = imgnew, gtnew, masknew
        if self.scale:
            return torch.tensor(img.copy()).permute(2, 0, 1).float(), torch.tensor(mask.copy()).permute(2, 0,1).float(), torch.tensor(
                gt.copy()).permute(2, 0, 1).float(), gtmax

        else:
            return torch.tensor(img.copy()).permute(2, 0, 1).float(), torch.tensor(mask.copy()).permute(2, 0, 1).float(), torch.tensor(gt.copy()).permute(2, 0, 1).float()



    def get_masked_img(self, img, gt, mask, idx):
        ratio1 = np.random.uniform(self.Lratio, self.ratio, 1)[0]  # np.random.uniform(0.0, 0.7, 1)[0]
        pick2set0 = random.sample([i for i in range(len(idx[0]))], int(len(idx[0]) * ratio1))
        pick2set0.sort()
        indexnew = (idx[0][pick2set0], idx[1][pick2set0], idx[2][pick2set0])
        img[indexnew] = self.assi

        # mask = np.where(gt > self.lowerb, 1, 0)
        if self.ori_size > self.target_size:
            img = img[:self.target_size, :self.target_size, ...]
            gt = gt[:self.target_size, :self.target_size, ...]
            mask = mask[:self.target_size, :self.target_size, ...]
        else:
            imgnew = np.zeros([self.target_size, self.target_size, 1])
            gtnew = np.zeros([self.target_size, self.target_size, 1])
            masknew = np.ones([self.target_size, self.target_size, 1])
            imgnew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = img
            gtnew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = gt
            masknew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = mask
            img, gt, mask = imgnew, gtnew, masknew

        return img, gt, mask



    def load_create_item_valuemask_reverse(self, index):
        # assi = 0.5#0#0.999
        gt_real = self.gtdataall[..., index]
        gt = self.dataall[..., index]

        if self.scale:
            gtmax = gt.max()
            gt = gt / gt.max()
        ratio2 = 1  # np.random.uniform(1, 1.5, 1)[0]# np.random.uniform(1, 4, 1)[0] # np.random.uniform(1, 4, 1)[0]#1#np.random.uniform(4, 5, 1)[0]
        gt = (gt * ratio2).clip(0, 1)

        # mask = np.where(gt > 0, 1, 0)
        # gt_re = np.where(gt == 0, self.assi, gt)
        mask = np.where(gt > self.lowerb, 1, 0)
        gt_re = np.where(gt <= self.lowerb, self.assi, gt)
        # idx = np.where((gt > 0))# & (gt < (0.9  * gt.max())))
        idx = np.where((gt_re != self.assi))
        img = np.copy(gt_re)


        ratio = np.random.uniform(self.Lratio, self.ratio, 1)[0]  # np.random.uniform(0.0, 0.7, 1)[0]
        # ratio2 = np.random.uniform(0.5, 1., 1)[0]
        pick2set0 = random.sample([i for i in range(len(idx[0]))], int(len(idx[0]) * ratio))
        pick2set0.sort()
        indexnew = (idx[0][pick2set0], idx[1][pick2set0], idx[2][pick2set0])
        img[indexnew] = self.assi  # 0
        # if self.ema:
        #     img2 = np.copy(gt_re)
        #     ratio = np.random.uniform(self.Lratio, self.ratio, 1)[0]
        #     pick2set0 = random.sample([i for i in range(len(idx[0]))], int(len(idx[0]) * ratio))
        #     pick2set0.sort()
        #     indexnew = (idx[0][pick2set0], idx[1][pick2set0], idx[2][pick2set0])
        #     img2[indexnew] = self.assi  # 0

        if self.ori_size > self.target_size:
            img = img[:self.target_size, :self.target_size, ...]
            gt = gt[:self.target_size, :self.target_size, ...]
            mask = mask[:self.target_size, :self.target_size, ...]
            # if self.ema:
            #     img2 = img2[:self.target_size, :self.target_size, ...]
        else:
            imgnew = np.zeros([self.target_size, self.target_size, 1]) + self.assi
            gtnew = np.zeros([self.target_size, self.target_size, 1])
            masknew = np.ones([self.target_size, self.target_size, 1])
            imgnew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = img
            gtnew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = gt
            masknew[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = mask
            # if self.ema:
            #     imgnew2 = np.zeros([self.target_size, self.target_size, 1]) + self.assi
            #     imgnew2[self.cut_ind:self.cut_ind + self.ori_size, self.cut_ind:self.cut_ind + self.ori_size, :] = img2
            #     img2 = imgnew2
            img, gt, mask = imgnew, gtnew, masknew
        # if self.scale:
        #     return torch.tensor(img.copy()).permute(2, 0, 1).float(), torch.tensor(mask.copy()).permute(2, 0, 1).float(), torch.tensor(
        #         gt.copy()).permute(2, 0, 1).float(), gtmax
        # else:
        #     if self.ema:
        #         return torch.tensor(img.copy()).permute(2, 0, 1).float(), torch.tensor(img2.copy()).permute(2, 0, 1).float(), torch.tensor(mask.copy()).permute(2, 0, 1).float(), torch.tensor(gt.copy()).permute(2, 0, 1).float()
        #     else:
        if self.for_val:
            return torch.tensor(img.copy()).permute(2, 0, 1).float(), torch.tensor(mask.copy()).permute(2, 0,
                                                                                                        1).float(), torch.tensor(
                gt.copy()).permute(2, 0, 1).float(), gt.max()

        else:
            return torch.tensor(img.copy()).permute(2, 0, 1).float(), torch.tensor(mask.copy()).permute(2, 0, 1).float(), torch.tensor(gt.copy()).permute(2, 0, 1).float()






