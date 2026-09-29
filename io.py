from pathlib import Path
import yaml, json, pandas as pd, numpy as np

def load_yaml(path):
    with open(path, encoding='utf-8') as f: return yaml.safe_load(f)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,default=str)

def load_dataset(path, label_col='label'):
    df=pd.read_csv(path)
    if label_col not in df.columns: raise ValueError(f"Missing target column: {label_col}")
    y=df[label_col].to_numpy()
    feature_df=df.drop(columns=[label_col])
    numeric=feature_df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0: raise ValueError('No numeric feature columns found.')
    X=numeric.to_numpy(dtype=np.float32)
    classes, y_enc=np.unique(y, return_inverse=True)
    if len(classes)!=2: raise ValueError(f'Binary classification required; found {len(classes)} classes.')
    return X, y_enc.astype(np.int32), classes.tolist(), list(numeric.columns)
