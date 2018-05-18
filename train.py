import os
import time
import torch
import cv2 as cv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from fcn_xu import fcn_xu,fcn_xu_dilated
from data_loader_13 import MR13RGBloader_CV
from metrics import runningScore
from loss import cross_entropy2d,weighted_loss,dice_loss
#from mpl.mpl import MaxPoolingLoss

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    #torch.manual_seed(1337)
    data_path='../DATA/MRBrainS13DataNii/'
    print(args)
    t_loader=MR13RGBloader_CV(root=data_path,val_num=args.val_num,is_val=False,is_transform=True,is_rotate=True,is_crop=True,is_histeq=True,forest=args.num_forest)
    print('train set T1 mean=',t_loader.T1mean)
    v_loader=MR13RGBloader_CV(root=data_path,val_num=args.val_num,is_val=True,is_transform=True,is_rotate=False,is_crop=True,is_histeq=True,forest=args.num_forest)
    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=1, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=1, num_workers=1,shuffle=False)
    # Setup Metrics
    running_metrics_single = runningScore(n_classes)
    running_metrics_single_test = runningScore(4)
    # Setup Model
    model = fcn_xu(n_classes=n_classes)
    vgg16 = models.vgg16(pretrained=True)
    model.init_vgg16_params(vgg16)
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    loss_ce = cross_entropy2d
    loss_dc = dice_loss
    best_iou=-100.0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_iou=checkpoint['best_iou']
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {}), best_iou={}"
                  .format(args.resume, checkpoint['epoch'],best_iou))
        else:
            best_iou=-100.0
            print("No checkpoint found at '{}'".format(args.resume))

    t = []
    loss_train=[]
    Dice_mean=[]
    Dice_CSF=[]
    Dice_GM=[]
    Dice_WM=[]
    t_pre=time.time()
    print('training prepared, cost {} seconds\n\n'.format(t_pre-t_begin))
    for epoch in range(args.n_epoch):
        t.append(epoch+1)
        model.train()
        adjust_learning_rate(optimizer,epoch)
        #loss_sum=0.0
        loss_epoch=0.0
        t_epoch=time.time()
        for i_train, (regions,T1s,IRs,T2s,lbls) in enumerate(trainloader):
            T1s,IRs,T2s,lbls=Variable(T1s.cuda()),Variable(IRs.cuda()),Variable(T2s.cuda()),Variable(lbls.cuda())
            optimizer.zero_grad()
            outputs=model(T1s)
            loss=loss_ce(input=outputs,target=lbls[:,int(args.num_forest/2),:,:])
                 #+loss_dc(input=outputs,target=lbls[:,int(args.num_forest/2),:,:])
            #loss_sum+=loss
            #if (i_train+1)%args.loss_avg==0:
            #    loss_sum/=args.loss_avg
            #    loss_sum.backward()
            #    optimizer.step()
            #    loss_sum=0.0
            loss.backward()
            optimizer.step()
            loss_epoch+=loss.data[0]

            if i_train==20:
                ax1=plt.subplot(231)
                ax1.imshow((T1s[0,1,:,:].data.cpu().numpy()*255+t_loader.T1mean).astype(np.uint8),cmap ='gray')
                ax1.set_title('train_img')
                ax1.axis('off')
                ax2=plt.subplot(232)
                ax2.imshow(t_loader.decode_segmap(lbls[0,1,:,:].data.cpu().numpy()).astype(np.uint8))
                ax2.set_title('train_label')
                ax2.axis('off')
                ax3=plt.subplot(233)
                model.eval()
                ax3.imshow(t_loader.decode_segmap(model(T1s)[0].data.max(0)[1].cpu().numpy()).astype(np.uint8))
                ax3.set_title('train_output')
                ax3.axis('off')
                #plt.tight_layout()
                #plt.subplots_adjust(wspace=0,hspace=.3)
                #plt.savefig('./fig_out/train_{}_20.png'.format(epoch+1))
                model.train()
        loss_epoch/=i_train
        loss_train.append(loss_epoch)
        t_train=time.time()
        print('epoch: ',epoch+1)
        print('--------------------------------Training--------------------------------')
        print('average loss in this epoch: ',loss_epoch)
        print('final loss in this epoch: ',loss.data[0])
        print('cost {} seconds up to now'.format(t_train-t_begin))
        print('cost {} seconds in this train epoch'.format(t_train-t_epoch))

        model.eval()
        for i_val, (regions_val,T1s_val,IRs_val,T2s_val,lbls_val) in enumerate(valloader):
            T1s_val,IRs_val,T2s_val=Variable(T1s_val.cuda(), volatile=True),Variable(IRs_val.cuda(), volatile=True),Variable(T2s_val.cuda(), volatile=True)
            outputs_single=model(T1s_val)[0,:,:,:]
            pred_single=outputs_single.data.max(0)[1].cpu().numpy()
            pred_single_test=np.zeros((pred_single.shape[0],pred_single.shape[1]),np.uint8)
            pred_single_test=v_loader.lbl_totest(pred_single)

            gt = lbls_val[0][int(args.num_forest/2)].numpy()
            gt_test=np.zeros((gt.shape[0],gt.shape[1]),np.uint8)
            gt_test=v_loader.lbl_totest(gt)
            running_metrics_single.update(gt, pred_single)
            running_metrics_single_test.update(gt_test, pred_single_test)

            if i_val==20:
                ax4=plt.subplot(234)
                ax4.imshow((T1s_val[0,1,:,:].data.cpu().numpy()*255+t_loader.T1mean).astype(np.uint8),cmap ='gray')
                ax4.set_title('src_img')
                ax4.axis('off')
                ax5=plt.subplot(235)
                ax5.imshow(gt_test.astype(np.uint8))
                ax5.set_title('gt')
                ax5.axis('off')
                ax6=plt.subplot(236)
                ax6.imshow(pred_single_test.astype(np.uint8))
                ax6.set_title('pred_single')
                ax6.axis('off')
                plt.tight_layout()
                plt.subplots_adjust(wspace=0,hspace=.3)
                plt.savefig('./fig_out/train_valid_{}_20.png'.format(epoch+1))
        score_single, class_iou_single = running_metrics_single.get_scores()
        score_single_test, class_iou_single_test = running_metrics_single_test.get_scores()
        Dice_mean.append(score_single_test['Mean Dice : \t'])
        Dice_CSF.append(score_single_test['Dice : \t'][1])
        Dice_GM.append(score_single_test['Dice : \t'][2])
        Dice_WM.append(score_single_test['Dice : \t'][3])
        print('--------------------------------All tissues--------------------------------')
        print('Back: Background,')
        print('GM: Cortical GM(red), Basal ganglia(green),')
        print('WM: WM(yellow), WM lesions(blue),')
        print('CSF: CSF(pink), Ventricles(light blue),')
        print('Back: Cerebellum(white), Brainstem(dark red)')
        #print('forest predict: ')
        #for k, v in score.items():
        #    print(k, v)
        print('single predict: ')
        for k, v in score_single.items():
            print(k, v)
        print('--------------------------------Only tests--------------------------------')
        print('tissue : Back , CSF , GM , WM')
        #print('forest predict: ')
        #for k, v in score_test.items():
        #    print(k, v)
        print('single predict: ')
        for k, v in score_single_test.items():
            print(k, v)
        t_test=time.time()
        print('cost {} seconds up to now'.format(t_test-t_begin))
        print('cost {} seconds in this validation epoch'.format(t_test-t_train))

        if score_single_test['Mean Dice : \t'] >= best_iou:
            best_iou = score_single_test['Mean Dice : \t']
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),
                     'best_iou':best_iou}
            torch.save(state, "{}_{}_val{}.pkl".format('FCN', 'MR13',str(args.val_num)))
            print('model saved!!!')
        ax1=plt.subplot(211)
        ax1.plot(t,loss_train,'g')
        ax1.set_title('train loss')
        ax2=plt.subplot(212)
        ax2.plot(t,Dice_mean,'k')
        ax2.plot(t,Dice_CSF,'r')
        ax2.plot(t,Dice_GM,'g')
        ax2.plot(t,Dice_WM,'b')
        ax2.set_title('validate Dice, R/G/B for CSF/GM/WM')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0,hspace=.3)
        plt.savefig('./fig_out/curve_FCN_val{}.png'.format(str(args.val_num)))
        running_metrics_single.reset()
        running_metrics_single_test.reset()
        print('\n\n')

if __name__ == '__main__':
    t_begin=time.time()
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--gpu_id', nargs='?', type=int, default=-1,
                        help='GPU id, -1 for cpu')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=2000,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--loss_avg', nargs='?', type=int, default=1,
                        help='loss average')
    parser.add_argument('--num_forest', nargs='?', type=int, default=3,
                        help='number of forest')
    parser.add_argument('--lr', nargs='?', type=float, default=1e-4,
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--val_num', nargs='?', type=int, default=5,
                        help='which set is left for validation')
    args = parser.parse_args()
    train(args)
