import argparse
import numpy as np
from pathlib import Path
from src.utils.io import load_dataset,load_yaml
from src.baselines.feature_selection import GeneticFeatureSelector,ParticleSwarmFeatureSelector
p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--data',required=True); p.add_argument('--config',required=True); a=p.parse_args(); c=load_yaml(a.config); X,y,_,_=load_dataset(a.data,c['label_column']); out=Path('results')/a.dataset/'feature_selection'; out.mkdir(parents=True,exist_ok=True)
for name,sel in [('GA',GeneticFeatureSelector(50,.8,.1,100,c['seed'])),('PSO',ParticleSwarmFeatureSelector(50,2,2,.7,100,c['seed']))]:
    sel.fit(X,y); np.save(out/f'{name.lower()}_mask.npy',sel.mask); print(name,'fitness=',sel.best_score,'selected_features=',int(sel.mask.sum()))
