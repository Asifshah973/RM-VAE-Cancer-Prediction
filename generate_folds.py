import argparse
from pathlib import Path
from src.utils.io import load_dataset,load_yaml
from src.preprocessing.folds import make_folds,save_folds
from src.utils.seed import set_seed
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--data',required=True); p.add_argument('--config',required=True); a=p.parse_args()
c=load_yaml(a.config); set_seed(c['seed']); X,y,classes,features=load_dataset(a.data,c['label_column']); folds=make_folds(y,c['cv']['n_splits'],c['seed']); save_folds(folds,Path('folds')/a.dataset); print(f'{a.dataset}: {len(y)} samples, {X.shape[1]} features, {len(folds)} folds')
