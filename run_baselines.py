import argparse
from pathlib import Path
import pandas as pd
from src.utils.io import load_dataset,load_yaml
from src.utils.seed import set_seed
from src.preprocessing.folds import make_folds
from src.baselines.classical import models
from src.evaluation.metrics import binary_metrics
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--data',required=True); p.add_argument('--config',required=True); a=p.parse_args()
c=load_yaml(a.config); set_seed(c['seed']); X,y,_,_=load_dataset(a.data,c['label_column']); folds=make_folds(y,c['cv']['n_splits'],c['seed']); out=Path('results')/a.dataset; out.mkdir(parents=True,exist_ok=True); rows=[]
for name,base in models(c['seed']).items():
  for fold,(tr,te) in enumerate(folds):
    base.fit(X[tr],y[tr]); proba=base.predict_proba(X[te])[:,1] if hasattr(base,'predict_proba') else base.predict(X[te]); m=binary_metrics(y[te],proba); m.update({'fold':fold,'method':name}); rows.append(m)
    print(name,fold+1,m['accuracy'])
new=pd.DataFrame(rows); path=out/'baseline_fold_results.csv'; new.to_csv(path,index=False); print(new.groupby('method')['accuracy'].agg(['mean','std']))
