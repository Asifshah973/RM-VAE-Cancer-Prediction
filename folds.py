from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

def make_folds(y,n_splits=10,seed=1234):
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    return [(tr,te) for tr,te in skf.split(np.zeros(len(y)),y)]

def save_folds(folds,outdir):
    Path(outdir).mkdir(parents=True,exist_ok=True)
    for i,(tr,te) in enumerate(folds):
        np.savez(Path(outdir)/f'fold_{i:02d}.npz',train_idx=tr,test_idx=te)
