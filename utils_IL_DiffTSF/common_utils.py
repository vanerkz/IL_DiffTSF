import numpy as np

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    x,r = pred.shape
    mask = true.flatten() <0.000001
    newtrue = true.flatten()
    newpred = pred.flatten()
    newtrue[mask] =1
    newpred[mask] =1
    newtrue= newtrue.reshape(-1,r)
    newpred= newpred.reshape(-1,r)
    return np.mean(np.abs((newpred - newtrue) / newtrue)), np.std(np.mean(np.abs((newpred - newtrue)) / newtrue,axis=1))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def commonmetric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mspe = MSPE(pred, true)
    shape = pred.shape
    return mae,mse,rmse,mspe,shape
