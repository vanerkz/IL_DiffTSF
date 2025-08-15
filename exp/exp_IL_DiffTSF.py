from data.data_loader import Dataset_ETT_hour,Dataset_ETT_day,Dataset_ETT_minute
from exp.exp_basic import Exp_Basic
from models.model import IL_DiffTSF, Estimator
from utils_IL_DiffTSF.tools import EarlyStopping,adjust_learning_rate
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.notebook import tqdm
import os
import time
from utils_IL_DiffTSF.common_utils import commonmetric
from utils_IL_DiffTSF.tools import loss_fn_sigma,loss_fn
from  utils_IL_DiffTSF.crps import crps_gaussian,crps_ensemble
import warnings
warnings.filterwarnings('ignore')

class Exp_IL_DiffTSF(Exp_Basic):
    def __init__(self, args):
        super(Exp_IL_DiffTSF, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'IL_DiffTSF':IL_DiffTSF,
        }
        if self.args.model=='IL_DiffTSF':
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.c_out, 
                self.args.label_len,
                self.args.pred_len,
                self.args.n_times,
                self.args.offset,
                self.args.d_model, 
                self.args.freq,
                self.device
            ).float()

            estimatormodel = Estimator(
                self.args.enc_in,
                self.args.c_out, 
                self.args.label_len,
                self.args.pred_len,
                self.args.freq, 
                self.args.d_model, 
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            estimatormodel = nn.DataParallel(estimatormodel, device_ids=self.args.device_ids)
        print("Trainable Parameters:"+str(sum(p.numel() for p in model.parameters())))
        print("Trainable Parameters:"+str(sum(p.numel() for p in estimatormodel.parameters())))
        return model,estimatormodel

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'exchange':Dataset_ETT_day,
            'weather':Dataset_ETT_hour,
            'electrans':Dataset_ETT_hour,
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm2':Dataset_ETT_minute,
        }
        Data = data_dict[self.args.data]
        timeenc = 1

        if flag == 'test':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; freq=args.freq
        elif flag =='val':
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size; freq=args.freq
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size; freq=args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True,       
            persistent_workers=True)

        return data_set, data_loader

    def _select_optimizer(self):

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        estimatormodel_optim = optim.Adam(self.estimatormodel.parameters(), lr=self.args.learning_rate)
        return model_optim,estimatormodel_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss() 
        return criterion

    def vali(self, vali_data, vali_loader,flag,denoisebool):
        self.model.eval()
        self.estimatormodel.eval()
        datalosslist=[]
        reglosslist=[]
        total_loss = []
        total_loss2 = []
        train_reconloss=[]
        crps_total=[]
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,_) in enumerate(tqdm(vali_loader)):
            pred, true,epsilon, pred_epsilon, x_zeros,_,matrixout,nonnmatrix,m2w= self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,flag)
            if denoisebool:
                sigmaout=self.estimatormodel(nonnmatrix.detach(),batch_y_mark.float().to(self.device))
            if flag != 'test':
                loss,dataloss,regloss=loss_fn_sigma(epsilon,pred_epsilon,m2w,self.args.reg)
                total_loss.append(loss.item())
                reglosslist.append(regloss.item())
                datalosslist.append(dataloss.item())
                lossmse=nn.MSELoss()
                train_reconloss.append(lossmse(x_zeros,true[:,-self.args.pred_len:,:self.args.c_out]).item())
                if denoisebool:
                    loss2=loss_fn(sigmaout,true[:,:,:self.args.c_out],x_zeros.detach(),self.args.pred_len)
                    total_loss2.append(loss2.item())
        if flag != 'test':   
            total_loss = np.mean(total_loss)
            total_loss2 = np.mean(total_loss2)
            datalosslist = np.mean(datalosslist)
            reglosslist = np.mean(reglosslist)
            train_reconloss = np.mean(train_reconloss)
        return total_loss,total_loss2,crps_total,datalosslist,reglosslist,train_reconloss

    def retdataloader(self,flag):
        return self._get_data(flag = 'train')
    
    def train(self, setting):
        kstep= int(32/self.args.batch_size)
        _, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        path = self.args.checkpoints
        best_model_path = os.path.join(path,setting+'.pth')
        best_model_pathest = os.path.join(path,setting+'est.pth')
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        early_stopping2 = EarlyStopping(patience=self.args.patience, verbose=True)
        train1=True
        if os.path.exists(best_model_path):
            print("load:",setting)
            self.model.load_state_dict(torch.load(best_model_path))
        else:
             print("No File, Train new")
        
        if os.path.exists(best_model_pathest):
            print("load est:",setting)
            self.estimatormodel.load_state_dict(torch.load(best_model_pathest))
        else:
             print("No EST File, Train new")
           
        time_now = time.time()
        train_steps = len(train_loader)
        
        model_optim,estimatormodel_optim= self._select_optimizer()
        
        self.model.train()
        self.estimatormodel.train()
        epoch=0
        tepochloss=[]
        tdataloss=[]
        tregloss=[]
        tvalepochloss=[]
        tvaldataloss=[]
        tvalregloss=[]
        tvalreconloss=[]
        ttrain_reconloss=[]
        model_optim.zero_grad()
        estimatormodel_optim.zero_grad()
        iter_count=0
        while(epoch<self.args.train_epochs):
            train_loss = []
            data_loss = []
            reg_loss = []
            train_reconloss=[]
            res_list = None
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,_) in enumerate(tqdm(train_loader)):
                iter_count += 1
                if train1:
                    self.model.train()
                    state='train'
                else:
                    self.model.eval()
                    state='val'
                pred, true,epsilon, pred_epsilon, x_zeros,_,matrixout,nonnmatrix,m2w= self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,state)
                
                if train1:
                    loss,dataloss,regloss=loss_fn_sigma(epsilon,pred_epsilon,m2w,self.args.reg)
                    loss.backward()
                    if (i+1) % kstep == 0 or (i+1) == len(train_loader):
                        model_optim.step()  # update the weights only after accumulating k small batches
                        model_optim.zero_grad()  # reset gradients for accumulation for the next large_batch
                    loss=loss.item()
                    train_loss.append(loss)
                    data_loss.append(dataloss.item())
                    reg_loss.append(regloss.item())
                    lossmse=nn.MSELoss()
                    train_reconloss.append(lossmse(x_zeros,true[:,-self.args.pred_len:,:self.args.c_out]).item())
                else:
                    self.estimatormodel.train()
                    sigmaout=self.estimatormodel(nonnmatrix.detach(),batch_y_mark.float().to(self.device))
                    loss2=loss_fn(sigmaout,true[:,:,:self.args.c_out],x_zeros.detach(),self.args.pred_len)
                    residual=true[:,-self.args.pred_len:,:self.args.c_out]-x_zeros.detach()
                    if res_list is None:
                        res_list=residual.detach().cpu().numpy()
                    else:
                        res_list=np.concatenate((res_list,residual.detach().cpu().numpy()),axis=0)
                    
                    loss2.backward()
                    if (i+1) % kstep == 0 or (i+1) == len(train_loader):
                        estimatormodel_optim.step()  # update the weights only after accumulating k small batches
                        estimatormodel_optim.zero_grad()  # reset gradients for accumulation for the next large_batch
            
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss))
            speed = (time.time()-time_now)/iter_count
            left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            train_loss = np.average(train_loss)
            data_loss = np.average(data_loss)
            reg_loss = np.average(reg_loss)
            train_reconloss = np.average(train_reconloss)
            tepochloss.append(train_loss)
            tdataloss.append(data_loss)
            tregloss.append(reg_loss)
            ttrain_reconloss.append(train_reconloss)

            vali_loss,vali_loss2,crps_total,valdatalosslist,valreglosslist,valreconloss = self.vali(vali_data, vali_loader,"val",not train1)
            tvalepochloss.append(vali_loss)
            tvaldataloss.append(valdatalosslist)
            tvalregloss.append(valreglosslist)
            tvalreconloss.append(valreconloss)

            if train1==True:
                early_stopping(vali_loss, self.model, best_model_path)
                train1 =not early_stopping.early_stop 
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            elif self.args.train_est:
                early_stopping2(vali_loss2,self.estimatormodel, best_model_pathest)
                adjust_learning_rate(estimatormodel_optim, epoch + 1, self.args)

            if early_stopping.early_stop and (early_stopping2.early_stop or not self.args.train_est):
                print("Early stopping")
                break
            epoch+=1
        
        self.model.load_state_dict(torch.load(best_model_path))
        try:
            self.estimatormodel.load_state_dict(torch.load(best_model_pathest))
        except:
            print("no est")
        
        return self.model,tepochloss,tdataloss,tregloss,tvalepochloss,tvaldataloss,tvalregloss,ttrain_reconloss,tvalreconloss


    def test(self, setting,evaluate=False):
        _, test_loader = self._get_data(flag='test')
        self.model.eval()
        path = self.args.checkpoints
        best_model_path = os.path.join(path,setting+'.pth')
        if evaluate:
            if os.path.exists(best_model_path):
                print("load:",setting)
                self.model.load_state_dict(torch.load(best_model_path))
                best_model_pathest = os.path.join(path,setting+'est.pth')
                if os.path.exists(best_model_pathest):
                    self.estimatormodel.load_state_dict(torch.load(best_model_pathest))
                else:
                    print("No model for Stage 2")
            else:
                print("No model for Stage 1")
        preds = None
        trues = None
        sigmaouts = None
        crps_total=None
        trues_full = None
        data_stamp=None
        sigmaouttopfull=None
        sigmaoutbtmfull=None
        sample=self.args.sampling
        predsamplefull=None
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,data_stamp_batch) in enumerate(tqdm(test_loader)):
            """if i==100:
                break"""
            predsample=None
            if sample:
                for i in tqdm(range(self.args.sampling_times),leave=False):
                    pred, true,_, _, _,batch_yorg,_,nonnmatrix,_= self._process_one_batch(
                        batch_x, batch_y, batch_x_mark, batch_y_mark,"test")
                    pred=pred.unsqueeze(0)
                    if predsample is None:
                        predsample=pred.detach().cpu().numpy()
                    else:
                        predsample=np.concatenate((predsample,pred.detach().cpu().numpy()),axis=0)
      
                pred=np.mean(predsample[:,:,-self.args.pred_len:,:], axis=0)
                if sigmaoutbtmfull is None:
                    sigmaouttopfull=np.percentile(predsample[:,:,-self.args.pred_len:,:],95, axis=0)
                    sigmaoutbtmfull=np.percentile(predsample[:,:,-self.args.pred_len:,:],10,axis=0)
                else:
                    sigmaouttopfull=np.concatenate((sigmaouttopfull,np.percentile(predsample[:,:,-self.args.pred_len:,:],95, axis=0)),axis=0)
                    sigmaoutbtmfull=np.concatenate((sigmaoutbtmfull,np.percentile(predsample[:,:,-self.args.pred_len:,:],10,axis=0)),axis=0)
            else:

                pred, true,_, _, _,batch_yorg,_,nonnmatrix, _= self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark,"test")
                sigmaout=self.estimatormodel(nonnmatrix,batch_y_mark.float().to(self.device))
                sigma=sigmaout[:,:,:]
                if sigmaouts is None:
                    sigmaouts=sigma.detach().cpu().numpy()
                else:
                    sigmaouts=np.concatenate((sigmaouts,sigma.detach().cpu().numpy()),axis=0)
                pred=pred[:,-self.args.pred_len:,:]
            true=true[:,-self.args.pred_len:,:self.args.c_out]
            if data_stamp is None:
                data_stamp=data_stamp_batch.detach().cpu().numpy()
            else:
                data_stamp=np.concatenate((data_stamp,data_stamp_batch.detach().cpu().numpy()),axis=0)
            
            if not sample:
                if preds is None:
                    preds=pred.detach().cpu().numpy()
                else:
                    preds=np.concatenate((preds,pred.detach().cpu().numpy()),axis=0)
            else:
                if preds is None:
                    preds=pred
                    predsamplefull=predsample
                else:
                    preds=np.concatenate((preds,pred),axis=0)
                    predsamplefull=np.concatenate((predsamplefull,predsample),axis=1)
            if trues is None:
                trues=true.detach().cpu().numpy()
                trues_full=batch_yorg.detach().cpu().numpy()
            else:
                trues=np.concatenate((trues,true.detach().cpu().numpy()),axis=0)
                trues_full=np.concatenate((trues_full,batch_yorg.detach().cpu().numpy()),axis=0)

        print("trues shape:",trues.shape)    
        if not sample:
            print("preds shape:",preds.shape)
            print("sigmaouts shape:",sigmaouts.shape)
            crps_total=crps_gaussian(trues,preds,sigmaouts)
            sigmaouts = sigmaouts.reshape(-1, sigmaouts.shape[-2],sigmaouts.shape[-1])
        else:
            print("sample shape:",predsamplefull.shape)
            crps_total=crps_ensemble(trues,predsamplefull,axis=0)

        crpsret=crps_total
        print("test_CRPS_mean:"+str(np.mean(np.array(crps_total))),"test_CRPS_var:"+str(np.var(np.array(crps_total))))
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        trues_full = trues_full.reshape(-1, trues_full.shape[-2], trues_full.shape[-1])
        sqtrue=trues.squeeze()
        sqpred=preds.squeeze()
        resultout = commonmetric(sqpred,sqtrue)
        return resultout,crpsret
    

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark,flag):
      batch_yin=torch.tensor(batch_y)
      batch_yorg=torch.tensor(batch_y)
      if flag =='test':
        batch_yin[:,-self.args.pred_len:,:]=torch.zeros_like(batch_y[:,-self.args.pred_len:,:])
      batch_x = batch_x.float().to(self.device)
      batch_y = batch_y.float()
      batch_yin = batch_yin.float()
      batch_x_mark = batch_x_mark.float().to(self.device)
      batch_y_mark = batch_y_mark.float().to(self.device)

      batch_y = batch_y.float().to(self.device)
      batch_yin = batch_yin.float().to(self.device)
      output,epsilon, pred_epsilon,x_zeros,matrixout,nonnmatrix,m2w= self.model(batch_yin, batch_y_mark,flag)
      batch_y = torch.concat((batch_y,batch_y_mark),2).to(self.device)
      
      return  output,batch_y,epsilon, pred_epsilon,x_zeros,batch_yorg,matrixout,nonnmatrix,m2w

        
