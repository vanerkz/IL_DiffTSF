import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils_NRU_RBN.tools import StandardScaler
from utils_NRU_RBN.timefeatures import time_features
import warnings
import math
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info

        self.label_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.flag=flag
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.features=='S':
            df_data =df_raw.iloc[:,-1:]
        elif self.features=='MS'or self.features=='M':
            df_data =df_raw.iloc[:,1:]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_stamp_raw=pd.to_datetime(df_stamp.date).values.view('int64').reshape(-1,1)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.set_type ==2 and self.features=='M':
            indexraw = np.arange(0,len(self.data_x) - self.label_len- self.pred_len  + 1,self.pred_len//4)
            s_begin = indexraw[index]
        else:
            s_begin = index

        s_end = s_begin + self.label_len 
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        date_stamp_raw=self.data_stamp_raw[s_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark,date_stamp_raw
    
    def __len__(self):
            if self.set_type ==2 and self.features=='M':
                out=len(np.arange(0,len(self.data_x)- self.label_len- self.pred_len + 1,self.pred_len//4))
            else:
                out=len(self.data_x) - self.label_len- self.pred_len + 1
            return out

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_ETT_day(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='D'):
        # size [seq_len, label_len, pred_len]
        # info
        """if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:"""
        self.label_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.flag=flag
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.features=='S':
            df_data =df_raw.iloc[:,-1:]
        elif self.features=='MS'or self.features=='M':
            df_data =df_raw.iloc[:,1:]
  
        if self.pred_len==336:
            scale=0.65
        else:
            scale=0.7
        num_train = int(len(df_raw) * scale)
        num_test = int(len(df_raw) * 0.2)
        
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)
        
        df_stamp = df_raw[['date']][border1:border2]
        
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        self.data_stamp_raw=pd.to_datetime(df_stamp.date).values.view('int64').reshape(-1,1)
        
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.set_type ==2 and self.features=='M':
            indexraw = np.arange(0,len(self.data_x) - self.label_len- self.pred_len  + 1,self.pred_len//4)
            s_begin = indexraw[index]
        else:
            s_begin = index
        s_begin = index
        s_end = s_begin + self.label_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        date_stamp_raw=self.data_stamp_raw[s_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,date_stamp_raw
    
    def __len__(self):
        if self.set_type ==2 and self.features=='M':
                out=len(np.arange(0,len(self.data_x) - self.label_len- self.pred_len + 1,self.pred_len//4))
        else:
                out=len(self.data_x) - self.label_len- self.pred_len + 1
        return out

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        self.label_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.flag=flag
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        if self.features=='S':
            df_data =df_raw.iloc[:,-1:]
        elif self.features=='MS'or self.features=='M':
            df_data =df_raw.iloc[:,1:]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)


        self.data_stamp_raw=pd.to_datetime(df_stamp.date).values.view('int64').reshape(-1,1)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):

        
        if self.set_type ==2 and self.features=='M':
            indexraw = np.arange(0,len(self.data_x) - self.label_len- self.pred_len  + 1,self.pred_len//4)
            s_begin = indexraw[index]
        else:
            s_begin = index
        s_end = s_begin + self.label_len 
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        date_stamp_raw=self.data_stamp_raw[s_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,date_stamp_raw
    
    def __len__(self):
            if self.set_type ==2 and self.features=='M':
                out=len(np.arange(0,len(self.data_x) - self.label_len- self.pred_len + 1,self.pred_len//4))
            else:
                out=len(self.data_x) - self.label_len- self.pred_len + 1
            return out
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)