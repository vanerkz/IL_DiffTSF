import numpy as np
import numpy as np
import matplotlib.pyplot as plt
#from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import pandas as pd
import seaborn as sns


def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def Corr(pred, true):
    """sig_p = np.std(pred, axis=0)
    sig_g = np.std(true, axis=0)
    m_p = pred.mean(0)
    m_g = true.mean(0)
    ind = (sig_g != 0)
    corr = ((pred - m_p) * (true - m_g)).mean(0) / (sig_p * sig_g)
    corr = (corr[ind]).mean()"""
    corr_list = []
    for index in range(len(pred)):
        predi =pd.Series(pred[index])
        truei =pd.Series(true[index])
        corr_list.append(predi.corr(truei))
    nparray=np.array(corr_list)
    df_data = pd.DataFrame(nparray)
    df_data = df_data.dropna()
    df_data = np.array(df_data)
    return np.mean(df_data)



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

"""def dtwmetrics(pred,true):
    d=0
    yscaled_type = np.array(true).astype(np.double)
    bscaled_type = np.array(pred).astype(np.double)
    for i in range(len(bscaled_type)):
        d = d+dtw.distance_fast(yscaled_type[i], bscaled_type[i])
    dtw_dis=d/len(bscaled_type)
    return dtw_dis"""

def metric(pred, true):

    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    #corr1 = CORR(pred, true)
    corr = Corr(pred, true)
    return mae,mse,rmse,mape,mspe,corr

def plotitems(preds,trues,items,top,auto):
    if top:
        res = preds-trues
        temp = np.argpartition(-res.sum(axis=1), 1)
        result_args = temp
        items=result_args[:4]
    if auto:
        autoplot(preds,trues,items)
    x,y=trues.shape
    fig = plt.figure(figsize=(8, 8))
    for plot_id, i in enumerate(items):
        subplots = [321, 322, 323, 324]
        ff, yy = preds[i], trues[i]
        ax = fig.add_subplot(subplots[plot_id])
        #plot_scatter(range(0, backcast_length), xx, color='b')
        ax.plot(range(0,y), yy, color='r',label="True")
        ax.scatter(range(0,y), yy, color='r')
        ax.plot(range(0,y),ff, color='g',label="Pred")
        ax.scatter(range(0,y),ff, color='g')
        ax.legend(loc='upper right')
    plt.show()


def res(pred,true):
    plt.clf()
    scaler=MinMaxScaler()
    yscaled=scaler.fit_transform(true)
    bscaled = scaler.transform(pred)
    res=bscaled-yscaled
    print("Mean:"+str(np.mean(res)))
    print("Skewness:"+str(stats.skew(res.flatten(), axis=0, bias=True)))
    df=pd.DataFrame(data={'res:':res.flatten()})
    sns.distplot(a=df, hist=True)
    plt.show()
    print(res.shape)

def commonmetric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mspe = MSPE(pred, true)
    shape = pred.shape
    return mae,mse,rmse,mspe,shape



"""# importing various package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# making Time series
spacing = np.linspace(-5 * np.pi, 5 * np.pi, num=100)
s = pd.Series(0.7 * np.random.rand(100) + 0.3 * np.sin(spacing))
 
# Creating Autocorrelation plot
x = pd.plotting.autocorrelation_plot(s)
 
# plotting the Curve
x.plot()
 
# Display
plt.show()"""