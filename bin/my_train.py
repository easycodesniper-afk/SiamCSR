import torch
import numpy as np
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
import sys
#sys.path.append('/XXX/SiamCSR')


from SiamCSR.config import config
from SiamCSR.rgbt_network import SiamCSRNet
from data.rgbt234_lasher import RGBT234_Lasher
from data.rgbt_dataset import RGBTDataset
from SiamCSR.transforms import ToTensor
from SiamCSR.loss import rpn_smoothL1, rpn_cross_entropy_balance
from SiamCSR.utils import adjust_learning_rate, freeze_former_3_layers, freeze_rgbt_layers, unfreeze_rgbt_layers


torch.manual_seed(config.seed)

def train(data_dir, resume_path=None, vis_port=None, init=None):

    #-----------------------
    rgbt_sequence = RGBT234_Lasher('/dataset/LasHeR3', list='gtot+alllasher.txt')
    print('rgbt_sequence', len(rgbt_sequence))
    rgbt_dataset = RGBTDataset(rgbt_sequence, ToTensor(), ToTensor(), ToTensor(), ToTensor(), name='gtot+lasher')
    anchors = rgbt_dataset.anchors
    rgbtloader = DataLoader(dataset=rgbt_dataset,
                            batch_size=config.train_batch_size,
                            shuffle=True, num_workers=config.train_num_workers,
                            pin_memory=True, drop_last=True)

    # create summary writer
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)
    # start training
    # -----------------------------------------------------------------------------------------------------#
    model = SiamCSRNet()
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,momentum=config.momentum, weight_decay=config.weight_decay)
  
    #load model weight
    # -----------------------------------------------------------------------------------------------------#
    start_epoch = 1
    if resume_path and init:
        print("init training with checkpoint %s" % resume_path + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(resume_path)
        if 'model' in checkpoint.keys():
            model.load_state_dict(checkpoint['model'])
        else:
            model_dict = model.state_dict()
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict)
        del checkpoint
        torch.cuda.empty_cache()
        print("inited checkpoint")
    elif resume_path and not init:
        print("loading checkpoint %s" % resume_path + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(resume_path)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            model.load_state_dict(checkpoint)

        del checkpoint
        torch.cuda.empty_cache()
        print("loaded checkpoint")
    elif not resume_path and config.pretrained_model:
        print("loading pretrained model %s" % config.pretrained_model + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(config.pretrained_model)
        if 'model' in checkpoint:
            origin_model_dict = checkpoint['model']
            model_dict = model.state_dict()
            model_dict.update(origin_model_dict)
            model.load_state_dict(model_dict)
        else :
        # change name and load parameters
            model_dict = model.state_dict()
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict)


    for epoch in range(start_epoch, config.epoch + 1):
        train_loss = []
        model.train()

        if epoch <= 20:
            freeze_rgbt_layers(model)
        else:
            if epoch == 21:
                unfreeze_rgbt_layers(model)
            freeze_former_3_layers(model)

        loss_temp_cls = 0
        loss_temp_reg = 0
        for i, data in enumerate(tqdm(rgbtloader)):
            exemplar_imgs_r, exemplar_imgs_t, instance_imgs_r, instance_imgs_t, \
            regression_target, conf_target = data
            regression_target, conf_target = regression_target.cuda(), conf_target.cuda()
            #pre_score=8,10,19,19 ； pre_regression=[8,20,19,19]
            pred_score_1, pred_score_2, pred_regression_1, pred_regression_2 = model(exemplar_imgs_r.cuda(), instance_imgs_r.cuda(), \
                                                exemplar_imgs_t.cuda(),instance_imgs_t.cuda())
            # [8, 5x19x19, 2]=[8,1805,2]
            pred_conf_1 = pred_score_1.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                     2,
                                                                                                                     1)
             #[8,5x19x19,4] =[8,1805,4]
            pred_offset_1 = pred_regression_1.reshape(-1, 4,
                                                  config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                     2,
                                                                                                                     1)
            cls_loss_1 = rpn_cross_entropy_balance(pred_conf_1, conf_target, config.num_pos, config.num_neg, anchors,
                                                 ohem_pos=config.ohem_pos, ohem_neg=config.ohem_neg)
            reg_loss_1 = rpn_smoothL1(pred_offset_1, regression_target, conf_target, config.num_pos, ohem=config.ohem_reg)

            pred_conf_2 = pred_score_1.reshape(-1, 2,
                                               config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                  2,
                                                                                                                  1)
            # [8,5x19x19,4] =[8,1805,4]
            pred_offset_2 = pred_regression_2.reshape(-1, 4,
                                                      config.anchor_num * config.score_size * config.score_size).permute(
                0,
                2,
                1)
            cls_loss_2 = rpn_cross_entropy_balance(pred_conf_2, conf_target, config.num_pos, config.num_neg, anchors,
                                                   ohem_pos=config.ohem_pos, ohem_neg=config.ohem_neg)
            reg_loss_2 = rpn_smoothL1(pred_offset_2, regression_target, conf_target, config.num_pos,
                                      ohem=config.ohem_reg)
            cls_loss = cls_loss_1 + cls_loss_2
            reg_loss = reg_loss_1 + reg_loss_2
            loss = cls_loss + config.lamb * reg_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)#config.clip=10 ，clip_grad_norm
            optimizer.step()

            step = (epoch - 1) * len(rgbtloader) + i
            summary_writer.add_scalar('train/cls_loss', cls_loss.data, step)
            summary_writer.add_scalar('train/reg_loss', reg_loss.data, step)
            train_loss.append(loss.detach().cpu())
            loss_temp_cls += cls_loss.detach().cpu().numpy()
            loss_temp_reg += reg_loss.detach().cpu().numpy()

            if (i + 1) % config.show_interval == 0:
            #if (i + 1) % 5 == 0:
                tqdm.write("[epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f lr: %.2e"
                           % (epoch, i, loss_temp_cls / config.show_interval, loss_temp_reg / config.show_interval,
                              optimizer.param_groups[0]['lr']))
                loss_temp_cls = 0
                loss_temp_reg = 0
        train_loss = np.mean(train_loss)

        valid_loss = []
        
        
        valid_loss=0

        print("EPOCH %d valid_loss: %.4f, train_loss: %.4f" % (epoch, valid_loss, train_loss))
        
        summary_writer.add_scalar('valid/loss',valid_loss, (epoch + 1) * len(rgbtloader))
        
        adjust_learning_rate(optimizer,config.gamma)  # adjust before save, and it will be epoch+1's lr when next load
       
        if epoch > 20 and epoch % config.save_interval == 0:
            if not os.path.exists('../models/'):
                os.makedirs("../models/")
            save_name = "../models/SiamCSR_234_dualsp_{}.pth".format(epoch)
            new_state_dict=model.state_dict()
            torch.save({
                'epoch': epoch,
                'model': new_state_dict,
                'optimizer': optimizer.state_dict(),
            }, save_name)
            print('save model: {}'.format(save_name))


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    
    # parameters
    parser=argparse.ArgumentParser(description="SiamCSR Train")

    parser.add_argument('--resume_path',default='', type=str, help="")

    parser.add_argument('--data',default='/dataset/LasHeR3',type=str,help=" the path of data")

    args=parser.parse_args()

    train(args.data, args.resume_path)  
