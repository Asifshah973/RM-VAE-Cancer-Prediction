"""Nested-CV driver.
For each outer fold, hyperparameters are selected only from the outer training data.
The expensive RM-VAE training is delegated to the same two-stage implementation.
"""
import argparse, itertools, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from src.utils.io import load_dataset,load_yaml
from src.utils.seed import set_seed
from src.preprocessing.normalization import fit_train_scaler,transform
from src.preprocessing.augmentation import gaussian_augment
from src.training.rmvae_trainer import RMVAETrainer
from src.training.dnn_trainer import DNNTrainer
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--data',required=True); p.add_argument('--config',required=True); a=p.parse_args(); c=load_yaml(a.config); set_seed(c['seed']); X,y,_,_=load_dataset(a.data,c['label_column']); out=Path('results')/a.dataset/'nested_cv'; out.mkdir(parents=True,exist_ok=True)
outer=StratifiedKFold(c['cv']['n_splits'],shuffle=True,random_state=c['seed']); grid=list(itertools.product([.1,.5,1.0],[.01,.1,1.0],[16,32,64,128])); records=[]
# This driver is intentionally conservative: reduce grid/epochs in a copy of the config for smoke tests.
for oi,(ot,oe) in enumerate(outer.split(X,y)):
  inner=StratifiedKFold(3,shuffle=True,random_state=c['seed']+oi); best=None
  for beta,lam,k in grid:
    scores=[]
    for it,(tr,va) in inner.split(X[ot],y[ot]):
      tr_idx=ot[tr]; va_idx=ot[va]; scaler=fit_train_scaler(X[tr_idx]); Xtr=transform(scaler,X[tr_idx]); Xva=transform(scaler,X[va_idx]); ytr=y[tr_idx]
      if c['augmentation']['enabled'] and a.dataset.lower()=='breast' and len(tr_idx)<c['augmentation']['n_threshold']: Xtr,ytr=gaussian_augment(Xtr,ytr,c['augmentation']['sigma'],c['seed']+oi)
      r=c['rmvae']; rt=RMVAETrainer(Xtr.shape[1],k,tuple(r['hidden_units']),tuple(r['decoder_units']),beta,lam,r['learning_rate'],r['batch_size'],r['epochs'],r['geometry_epsilon'],r['projection_dimension'],r['finite_difference_delta'],c['seed']+oi,r['geometry_batch_size']); rt.fit(Xtr); Ztr=rt.embeddings(Xtr); Zva=rt.embeddings(Xva); d=c['dnn']; dt=DNNTrainer(k,tuple(d['hidden_units']),d['dropout'],d['learning_rate'],d['batch_size'],d['epochs'],d['patience'],c['seed']+oi); dt.fit(Ztr,ytr); scores.append(accuracy_score(y[va_idx],dt.predict_proba(Zva)>=.5))
    score=float(np.mean(scores));
    if best is None or score>best['score']: best={'beta':beta,'lambda':lam,'k':k,'score':score}
  records.append({'outer_fold':oi,**best}); print(records[-1])
pd.DataFrame(records).to_csv(out/'selected_hyperparameters.csv',index=False)
