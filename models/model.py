import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import DataEmbedding,TemporalEmbedding
import numpy as np
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()
    
class ConvBlock(nn.Conv2d):
    """
        Conv2D Block
            Args:
                x: (N, C_in, H, W)
            Returns:
                y: (N, C_out, H, W)
    """

    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                    stride=1, padding='same', dilation=1, groups=1, bias=True, gn=False, gn_groups=1):
        
        if padding == 'same':
            padding = kernel_size // 2 * dilation

        super(ConvBlock, self).__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)

        self.activation_fn = nn.SiLU() if activation_fn else None
        self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None

    def forward(self, x, time_embedding=None,residual=False):
        if residual:
            # in the paper, diffusion timestep embedding was only applied to residual blocks of U
            x = super(ConvBlock, self).forward(x)

            y=x+time_embedding
        else:
            y = super(ConvBlock, self).forward(x)
        y = self.group_norm(y) if self.group_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y
        
        return y
    
class Denoiser(nn.Module):
    
    def __init__(self, image_resolution, hidden_dims=[256, 256], diffusion_time_embedding_dim = 256, n_times=1000):
        super(Denoiser, self).__init__()
        
        _, d_ff, img_C = image_resolution
        
        kernel_size=3
        self.time_embedding = SinusoidalPosEmb(diffusion_time_embedding_dim)
        
        self.in_project = ConvBlock(img_C, hidden_dims[0], kernel_size=1)

        self.convs = nn.ModuleList([ConvBlock(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=kernel_size)])
        for idx in range(1, len(hidden_dims)):
            self.convs.append(ConvBlock(hidden_dims[idx-1], hidden_dims[idx], kernel_size=kernel_size, dilation=kernel_size**((idx-1)//2),
                                                    activation_fn=True, gn=True, gn_groups=8))                                
                               
        self.out_project = ConvBlock(hidden_dims[-1], out_channels=1, kernel_size=1)
        
        
    def forward(self, perturbed_x, diffusion_timestep):
        y = perturbed_x
        diffusion_embedding = self.time_embedding(diffusion_timestep)
 
        diffusion_embedding = diffusion_embedding.unsqueeze(-1).unsqueeze(-2)
        y = self.in_project(y)
        for i in range(1, len(self.convs)):
            y=self.convs[i](y, diffusion_embedding,residual = True)

        y = self.out_project(y)


        return y
        
  
class Diffusion(nn.Module):
    def __init__(self, model, n_times, input_size,pred_len,label_len,d_ff,beta_minmax=[1e-4, 2e-2],freq='t', device='cuda'):
    
        super(Diffusion, self).__init__()
        freq_map = {'1H':4, '15min':5,'1D':3}
        timefeatureNos = freq_map[freq]
        self.label_len=label_len
        self.pred_len=pred_len
        self.input_size=input_size
        self.n_times = n_times
        self.model = model
        self.d_ff=d_ff
        self.device=device
        beta_1, beta_T = beta_minmax
        self.betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times).to(self.device)
        self.sqrt_betas = torch.sqrt(self.betas).to(self.device)
        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas).to(self.device)
        self.sqrt_alpha_bars = torch.sqrt(torch.cumprod(self.alphas, dim=0)).to(self.device)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - torch.cumprod(self.alphas, dim=0)).to(self.device)
        self.MLP2=nn.Linear(input_size, d_ff)
        nn.init.orthogonal_(self.MLP2.weight)
        
        self.enc_embedding = DataEmbedding(self.input_size-timefeatureNos, timefeatureNos, d_ff,self.n_times, "fixed", freq)
        
        hidden_dims = [d_ff * (2 ** i) for i in range(2)]  # automatically grows each layer

        modules = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            modules.extend(vanilla_block(in_dim, out_dim))  # extend because vanilla_block returns a list

        self.encodervar = nn.Sequential(*modules)
        kernel_size=3
        dilation=1
        paddingno = kernel_size // 2 * dilation
        self.decoder_input = nn.Conv1d(in_channels=self.label_len, out_channels=pred_len, stride=1, kernel_size=kernel_size,padding=paddingno,dilation=dilation)
        
        modules2 = []
        # Reverse hidden_dims to go from largest → smallest
        rev_dims = hidden_dims[::-1]

        for idx, (in_dim, out_dim) in enumerate(zip(rev_dims[:-1], rev_dims[1:])):
            # Last pair gets Tanh
            if idx == len(rev_dims) - 2:
                modules2.extend(vanilla_block(in_dim, out_dim, activation=nn.Tanh()))
            else:
                modules2.extend(vanilla_block(in_dim, out_dim))

        self.decodervar = nn.Sequential(*modules2)

    def make_noisy(self, x_zeros, t):
        epsilon = torch.randn_like(x_zeros).to(self.device)
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
        return noisy_sample, epsilon,sqrt_alpha_bar, sqrt_one_minus_alpha_bar
    
    
    def forward(self, x_zeros,flag):
        B,_,_=x_zeros.shape
        t = torch.randint(low=0, high=self.n_times-1, size=(B,)).long().to(self.device)
        
        inlabel=x_zeros
        label=inlabel[:,:self.label_len,:]
        if flag=='train':
            self.enc_embedding.train()
            for param in self.enc_embedding.parameters():
                param.requires_grad =True
            self.encodervar.train()
            for param in self.encodervar.parameters():
                param.requires_grad =True
            self.decoder_input.train()
            for param in self.decoder_input.parameters():
                param.requires_grad =True
            self.decodervar.train()
            for param in self.decodervar.parameters():
                param.requires_grad =True
            self.MLP2.train()
            for param in self.MLP2.parameters():
                param.requires_grad =True
        else:
            self.enc_embedding.eval()
            for param in self.enc_embedding.parameters():
                param.requires_grad =False
            self.encodervar.eval()
            for param in self.encodervar.parameters():
                param.requires_grad =False
            self.decoder_input.eval()
            for param in self.decoder_input.parameters():
                param.requires_grad =False
            self.decodervar.eval()
            for param in self.decodervar.parameters():
                param.requires_grad =False
            self.MLP2.eval()
            for param in self.MLP2.parameters():
                param.requires_grad =False
        embedlabel2=self.enc_embedding(label,t.unsqueeze(1).repeat_interleave(self.label_len, dim=1))
        result = self.encodervar(embedlabel2)
        resaroutnew=self.decoder_input(result)
        context = self.decodervar(resaroutnew)
        m2w=self.MLP2.weight
        x_zeros=self.MLP2(inlabel[:,-self.pred_len:,:])
        x_zeros=torch.tanh(x_zeros)
        perturbed_images,epsilon, sqrt_alpha_bar, sqrt_one_minus_alpha_bar  = self.make_noisy(x_zeros, t)
    

        if flag=='train':
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad =True
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad =False
        pred_epsilon = self.model(torch.concat((perturbed_images.unsqueeze(1),context.unsqueeze(1)),1),t)
        nonnmatrix=(perturbed_images-(pred_epsilon.squeeze(1)*sqrt_one_minus_alpha_bar))/sqrt_alpha_bar
        nonnmatrix=nonnmatrix.clamp(-1 + 1e-6, 1 - 1e-6)
        pred_epsilon=pred_epsilon.squeeze(1)
        matrixout=perturbed_images.clamp(-1 + 1e-6, 1 - 1e-6)
        x_t=torch.atanh(nonnmatrix)
        w=self.MLP2.weight
        b=self.MLP2.bias
        x_0=x_t-b
        invw=torch.linalg.pinv(w).transpose(0,1)
        x_0=torch.matmul(x_0,invw)
        x_zerostemp=x_0
            
        return epsilon, pred_epsilon,x_zerostemp,matrixout,nonnmatrix,m2w


    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def denoise_at_t(self, t,tt,x_t,context):
        if tt > 1:
            z = torch.randn_like(x_t).to(self.device)
        else:
            z = torch.zeros_like(x_t).to(self.device)

        self.model.eval()
        for param in self.model.parameters():
                param.requires_grad =False
        epsilon_pred = self.model(torch.concat((x_t.unsqueeze(1),context.unsqueeze(1)),1),t)
        x_t=x_t.squeeze(1)
        epsilon_pred=epsilon_pred.squeeze(1)

        alpha = self.extract(self.alphas, t, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, t, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas, t, x_t.shape)
    
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1-alpha)/sqrt_one_minus_alpha_bar*epsilon_pred) + sqrt_beta*z

        return x_t_minus_1.clamp(-1 + 1e-6 ,1 - 1e-6)
                
    def sample(self, N,inlabel):       
        x_t = torch.randn((N, self.pred_len, self.d_ff)).to(self.device)
        for t in range(self.n_times-1, -1, -1): 

            tt = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(self.device)
            label=inlabel[:,:self.label_len,:]
            embedlabel=self.enc_embedding(label,tt.unsqueeze(1).repeat_interleave(self.label_len, dim=1))
            result = self.encodervar(embedlabel)
            resaroutnew=self.decoder_input(result)
            context = self.decodervar(resaroutnew)
            x_t = self.denoise_at_t(tt,t,x_t,context)

        matrixout=x_t
        x_t=torch.atanh(x_t)
        w=self.MLP2.weight
        b=self.MLP2.bias
        x_0=x_t-b
        invw=torch.linalg.pinv(w).transpose(0,1)
        x_0=torch.matmul(x_0,invw)

        return x_0,matrixout

class IL_DiffTSF(nn.Module):
    def __init__(self, enc_in, c_out,label_len, out_len,n_times,offset, d_model=512, freq='h', device=torch.device('cuda:0')):
        super(IL_DiffTSF, self).__init__()

        self.d_model=d_model
        self.pred_len = out_len
        self.label_len=label_len
        self.c_out = c_out
        self.offset=offset
        self.enc_in=enc_in
        self.device=device
        self.n_times=n_times
        freq_map = {'1H':4, '15min':5,'1D':3}
        timefeatureNos = freq_map[freq]
        self.timefeatureNos=timefeatureNos
        beta_minmax=[1e-4, 2e-2] 
        n_layers =4
        hidden_dim = d_model
        hidden_dims = [hidden_dim for _ in range(n_layers)]
        
        self.model = Denoiser(image_resolution=(self.pred_len,d_model, 2),
                 hidden_dims=hidden_dims, 
                 diffusion_time_embedding_dim=hidden_dim,
                 n_times=self.n_times).to(device)
        self.diffusion = Diffusion(self.model,n_times=self.n_times, input_size=enc_in+timefeatureNos,pred_len=self.pred_len,label_len=self.label_len,d_ff=d_model, beta_minmax=beta_minmax,freq=freq, device=self.device).to(self.device)

    def forward(self,inlabel, x_mark_dec,flag):
        offset=self.offset
        if flag=='train' or flag== 'val' :
            batch_size = inlabel.shape[0]
            if self.timefeatureNos==0:
                x_enccat=inlabel
            else:
                x_enccat=torch.concat((inlabel,x_mark_dec),2)
        if flag=='train':
            epsilon, pred_epsilon,x_zeros,matrixout,nonnmatrix,m2w=self.diffusion(x_enccat,"train")
            outputs=0
            if self.timefeatureNos==0:
                x_zeros=x_zeros
            else:
                x_zeros=x_zeros[:,:,:-self.timefeatureNos]
            if offset:
                ref=inlabel[:,-self.pred_len-1:-self.pred_len,:]
                pred=x_zeros[:,0:1,:]
                outputs=x_zeros+(ref-pred)
        elif flag=='val':
            with torch.no_grad():
                epsilon, pred_epsilon,x_zeros,matrixout,nonnmatrix,m2w=self.diffusion(x_enccat,"val")
                if self.timefeatureNos==0:
                    x_zeros=x_zeros
                else:
                    x_zeros=x_zeros[:,:,:-self.timefeatureNos]
            outputs=0
            if offset:
                ref=inlabel[:,-self.pred_len-1:-self.pred_len,:]
                pred=x_zeros[:,0:1,:]
                outputs=x_zeros+(ref-pred)
        else:
            batch_size = inlabel.shape[0] 
            if self.timefeatureNos==0:
                x_enccat=inlabel[:,:-self.pred_len,:]
            else:
                x_enccat=torch.concat((inlabel[:,:-self.pred_len,:],x_mark_dec[:,:self.label_len,:]),2)
            
            with torch.no_grad():
                sampleres,nonnmatrix=self.diffusion.sample(batch_size,x_enccat)
            if self.timefeatureNos==0:
                outputs=sampleres
            else:
                outputs=sampleres[:,:,:-self.timefeatureNos]
            
            if offset:
                ref=inlabel[:,-self.pred_len-1:-self.pred_len,:]
                pred=outputs[:,0:1,:]
                outputs=outputs+(ref-pred)
            x_zeros=0
            epsilon =0
            pred_epsilon=0
            matrixout=nonnmatrix
            m2w=0
        return outputs,epsilon, pred_epsilon,x_zeros,matrixout,nonnmatrix,m2w
    
def vanilla_block(in_feat, out_feat, activation=None):
    layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.SiLU() if activation is None else activation)

    return layers

class Estimator(nn.Module):
    def __init__(self, enc_in, c_out,label_len, out_len,freq, d_model=512,
                device=torch.device('cuda:0')):
        super(Estimator, self).__init__()

        self.d_model=d_model
        self.pred_len = out_len
        self.label_len=label_len
        self.c_out = c_out
        self.enc_in=enc_in
        self.device=device

        modules = []
        modules.append(
            nn.Sequential(
                    nn.Linear(d_model*2,d_model*2),
                    nn.ReLU(),
                    nn.Linear(d_model*2, enc_in),
                    nn.Softplus())
        )
        self.EST = nn.Sequential(*modules)
        self.position=TemporalEmbedding(d_model=d_model, embed_type='fixed', freq=freq)
        
    def forward(self, hiddenmatrix,temporalenc):
        temporalenc=self.position(temporalenc[:,-self.pred_len:])
        H_y0_w_C= torch.concat((hiddenmatrix,temporalenc),2)
        sigmaout=self.EST(H_y0_w_C)
        return sigmaout
    
