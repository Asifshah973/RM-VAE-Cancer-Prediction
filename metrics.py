import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error,roc_auc_score

def binary_metrics(y_true,proba):
    pred=(proba>=.5).astype(int)
    out={'accuracy':accuracy_score(y_true,pred),'precision':precision_score(y_true,pred,zero_division=0),'recall':recall_score(y_true,pred,zero_division=0),'f1':f1_score(y_true,pred,zero_division=0),'mse':mean_squared_error(y_true,proba),'rmse':mean_squared_error(y_true,proba)**.5}
    try: out['roc_auc']=roc_auc_score(y_true,proba)
    except ValueError: out['roc_auc']=float('nan')
    return out

def fold_summary(rows):
    import pandas as pd
    df=pd.DataFrame(rows); out={}
    for c in df.columns:
        if c in ('fold','method'): continue
        if np.issubdtype(df[c].dtype,np.number): out[c+'_mean']=df[c].mean(); out[c+'_sd']=df[c].std(ddof=1)
    return out
