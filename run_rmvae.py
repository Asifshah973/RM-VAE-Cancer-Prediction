import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.io import load_dataset,load_yaml,save_json
from src.utils.seed import set_seed
from src.preprocessing.folds import make_folds
from src.preprocessing.normalization import fit_train_scaler,transform
from src.preprocessing.augmentation import gaussian_augment
from src.training.rmvae_trainer import RMVAETrainer
from src.training.dnn_trainer import DNNTrainer
from src.evaluation.metrics import binary_metrics

p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--data',required=True); p.add_argument('--config',required=True); a=p.parse_args()
c=load_yaml(a.config); set_seed(c['seed']); X,y,classes,features=load_dataset(a.data,c['label_column']); folds=make_folds(y,c['cv']['n_splits'],c['seed']); out=Path('results')/a.dataset; out.mkdir(parents=True,exist_ok=True)
rows=[]; histories=[]
for fold,(tr,te) in enumerate(folds):
    set_seed(c['seed']+fold)
    scaler=fit_train_scaler(X[tr]); Xtr=transform(scaler,X[tr]); Xte=transform(scaler,X[te])
    ytr=y[tr].copy();
    if c['augmentation']['enabled'] and a.dataset.lower()=='breast' and len(tr)<c['augmentation']['n_threshold']:
        Xtr,ytr=gaussian_augment(Xtr,ytr,c['augmentation']['sigma'],c['seed']+fold)
    r=c['rmvae']; trainer=RMVAETrainer(Xtr.shape[1],r['latent_dim'],tuple(r['hidden_units']),tuple(r['decoder_units']),r['beta'],r['lambda'],r['learning_rate'],r['batch_size'],r['epochs'],r['geometry_epsilon'],r['projection_dimension'],r['finite_difference_delta'],c['seed']+fold,r['geometry_batch_size'])
    trainer.fit(Xtr); Ztr=trainer.embeddings(Xtr); Zte=trainer.embeddings(Xte)
    if len(np.unique(ytr))==2 and len(ytr)>=20:
        Zfit,Zval,yfit,yval=train_test_split(Ztr,ytr,test_size=.1,stratify=ytr,random_state=c['seed']+fold)
    else: Zfit,Zval,yfit,yval=Ztr,Ztr,ytr,ytr
    d=c['dnn']; dt=DNNTrainer(Ztr.shape[1],tuple(d['hidden_units']),d['dropout'],d['learning_rate'],d['batch_size'],d['epochs'],d['patience'],c['seed']+fold); dt.fit(Zfit,yfit,Zval,yval); proba=dt.predict_proba(Zte)
    m=binary_metrics(y[te],proba); m.update({'fold':fold,'method':'RM-VAE+DNN'}); rows.append(m); histories.extend([{'fold':fold,**h} for h in trainer.history])
    print(f'fold {fold+1}/{len(folds)} accuracy={m["accuracy"]:.4f}')
pd.DataFrame(rows).to_csv(out/'fold_results.csv',index=False); pd.DataFrame(histories).to_csv(out/'rmvae_history.csv',index=False); save_json({'classes':classes,'n_samples':len(y),'n_features':X.shape[1],'feature_names':features},out/'dataset_info.json')
print(pd.DataFrame(rows).mean(numeric_only=True).to_string())
