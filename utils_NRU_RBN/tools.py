import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torchmetrics.regression import PearsonCorrCoef
from torch.distributions import Normal, kl

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def minmax(x_t,newmin,newmax):
    v_min, v_max = x_t.min(), x_t.max()
    new_min, new_max = newmin, newmax
    x_t = (x_t - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    return x_t

def loss_quantlie(pmeanout,nmeanout,pred,labels):
    x,y,z=pmeanout.shape
    labels=labels[:,-y:,:1].squeeze()-pred[:,:,:1].squeeze()
    #mean=torch.mean(labels,dim=1).unsqueeze(1).repeat_interleave(y,dim=1)
    plabels=torch.tensor(labels)
    nlabels=torch.tensor(labels)
    pmax=plabels<=0
    nmax=nlabels>0
    plabels[pmax]=1000
    nlabels[nmax]=1000
    nlabels=torch.abs(nlabels)
    nmin=torch.min(nlabels,dim=1).values.unsqueeze(1).repeat_interleave(y,dim=1)
    pmin=torch.min(plabels,dim=1).values.unsqueeze(1).repeat_interleave(y,dim=1)
    pindex=(plabels>900).reshape(x,-1)
    nindex=(nlabels>900).reshape(x,-1)
    plabels[pindex]=pmin[pindex]
    nlabels[nindex]=nmin[nindex]
    #plabels[plabels ==1000]=0
    #nlabels[nlabels==1000]=0
    plabels=plabels.reshape(x,-1)
    nlabels=nlabels.reshape(x,-1)

    loss=0
    loss2=0
    quantiles = np.array(list(range(10,90)))/100
    num_ts = labels.shape[0]
    for q, rho in enumerate(quantiles):
        ypred_rho = pmeanout[:, :, q].view(num_ts, -1)
        loss += torch.max(rho * (ypred_rho-plabels),(rho-1) * (ypred_rho-plabels))

    for q, rho in enumerate(quantiles):
        ypred_rho = nmeanout[:, :, q].view(num_ts, -1)
        loss2 += torch.max(rho * (ypred_rho-nlabels),(rho-1) * (ypred_rho-nlabels))
    return loss.mean()+loss2.mean()



def loss_fn(sigmaout,labels,x_zeros,pred_len):
    sumraw=labels[:,-pred_len:,:]-x_zeros

    distribution = torch.distributions.normal.Normal(torch.zeros_like(sigmaout), sigmaout)
    likelihood = distribution.log_prob(sumraw)
    return -torch.mean(likelihood)

def loss_fn_sigma(epsilon, pred_epsilon,m2w,epoch,regvalue):
    loss = torch.nn.MSELoss()

    sym2=torch.mm(torch.t(m2w),m2w)
    sym2=sym2-torch.eye(torch.t(m2w).shape[0]).to(m2w.device)
    u, s, v = torch.svd(sym2)  # Singular value decomposition
    spectral_norm_difference = s[0]
    dataloss=loss(epsilon,pred_epsilon)
    regloss=regvalue*(spectral_norm_difference)

    distribution =dataloss+regloss
    
    return distribution, dataloss,regloss

class TorchStandardScaler:
  def fit(self, x):
    self.mean = x.mean(2, keepdim=True)
    self.std = x.std(2, unbiased=False, keepdim=True)
  def transform(self, x):
    x -= self.mean
    x /= (self.std)
    return x

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.patience == 0:
            self.best_score = None
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean