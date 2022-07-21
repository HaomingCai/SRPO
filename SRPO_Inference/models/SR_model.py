import logging
from collections import OrderedDict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss
import torchvision

import numpy as np

logger = logging.getLogger('base')

def fspecial_gauss(size, sigma, channels):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    g = torch.from_numpy(g/g.sum()).float().unsqueeze(0).unsqueeze(0)
    return g.repeat(channels,1,1,1)

def gaussian_filter(input, win):
    out = F.conv2d(input, win, stride=1, padding=0, groups=input.shape[1])
    return out

def ssim(X, Y, win, get_ssim_map=False, get_cs=False, get_weight=False):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    win = win.to(X.device)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(X * X, win) - mu1_sq
    sigma2_sq = gaussian_filter(Y * Y, win) - mu2_sq
    sigma12 = gaussian_filter(X * Y, win) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    cs_map = F.relu(cs_map)  # force the ssim response to be nonnegative to avoid negative results.
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_val = ssim_map.mean([1, 2, 3])

    if get_weight:
        weights = torch.log((1 + sigma1_sq / C2) * (1 + sigma2_sq / C2))
        return ssim_map, weights

    if get_ssim_map:
        return ssim_map

    if get_cs:
        return ssim_val, cs_map.mean([1, 2, 3])

    return ssim_val


class SSIM(torch.nn.Module):
    def __init__(self, channels=3):

        super(SSIM, self).__init__()
        self.win = fspecial_gauss(11, 1.5, channels)

    def forward(self, X, Y, as_loss=True):
        assert X.shape == Y.shape
        if as_loss:
            score = ssim(X, Y, win=self.win)
            return 1 - score.mean()
        else:
            with torch.no_grad():
                score = ssim(X, Y, win=self.win)
            return score



class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)


        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # Define upsample type for offset map in valid and train phase
        self.output_set = train_opt["output_set"]
        self.offset_up_mode_train = train_opt["offset_up_mode_train"]
        self.offset_up_mode_test = train_opt["offset_up_mode_test"]

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss ==================================================================
            self.loss_type1 = train_opt['pixel_criterion1']
            if self.loss_type1 == 'l1':
                self.cri_pix1 = nn.L1Loss().to(self.device)
            elif self.loss_type1 == 'l2':
                self.cri_pix1 = nn.MSELoss().to(self.device)
            elif self.loss_type1 == 'cb':
                self.cri_pix1 = CharbonnierLoss().to(self.device)
            elif self.loss_type1 == 'ssim':
                self.cri_pix1 = SSIM(channels=3).to(self.device)
            else:
                raise NotImplementedError('Loss 1 type [{:s}] is not recognized.')


            self.loss_type2 = train_opt['pixel_criterion2']
            if self.loss_type2  == 'l1':
                self.cri_pix2 = nn.L1Loss().to(self.device)
            elif self.loss_type2  == 'l2':
                self.cri_pix2 = nn.MSELoss().to(self.device)
            elif self.loss_type2  == 'cb':
                self.cri_pix2 = CharbonnierLoss().to(self.device)
            elif self.loss_type2  == 'ssim':
                self.cri_pix2 = SSIM(channels=3).to(self.device)
            else:
                raise NotImplementedError('Loss 2 type [{:s}] is not recognized.')

            self.l_pix1_w = train_opt['pixel1_weight']
            self.l_pix2_w = train_opt['pixel2_weight']
            self.l_pix3_w = train_opt['pixel3_weight']
            # loss ==================================================================


            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()



    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT


    def optimize_parameters(self, step):

        self.optimizer_G.zero_grad()

        '''Generate OffsetSR and Final SR'''
        self.offset_sr, self.fake_H = self.netG(self.var_L, self.offset_up_mode_train)

        '''
        The optimization can be found in our publication as well.
        We cannot release this part as well.
        '''
        l_pix = None

        l_pix.backward()
        self.optimizer_G.step()
        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            _, self.fake_H = self.netG(self.var_L, self.offset_up_mode_test)
            self.offset_H  = self.fake_H[:, 0:3 , :, :] 
            self.offset    = self.fake_H[:, 3:  , :, :] 
        self.netG.train()


    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        out_dict['offset_H'] = self.offset_H.detach()[0].float().cpu()
        out_dict['offset'] = self.offset.detach()[0].float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
