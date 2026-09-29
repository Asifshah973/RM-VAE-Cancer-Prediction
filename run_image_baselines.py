import argparse, numpy as np, tensorflow as tf
from pathlib import Path
from src.baselines.image_models import build_vgg16,build_resnet50,compile_image_model
p=argparse.ArgumentParser(); p.add_argument('--images',required=True,help='NPY array [N,H,W,C]'); p.add_argument('--labels',required=True,help='NPY binary labels'); p.add_argument('--model',choices=['vgg16','resnet50'],required=True); p.add_argument('--epochs',type=int,default=100); p.add_argument('--seed',type=int,default=1234); a=p.parse_args(); tf.keras.utils.set_random_seed(a.seed); X=np.load(a.images).astype('float32'); y=np.load(a.labels).astype('int32');
if X.ndim!=4: raise ValueError('Images must have shape [N,H,W,C].'); model=build_vgg16(X.shape[1:]) if a.model=='vgg16' else build_resnet50(X.shape[1:]); compile_image_model(model,.001); model.fit(X,y,batch_size=32,epochs=a.epochs,verbose=1); Path('checkpoints').mkdir(exist_ok=True); model.save(f'checkpoints/{a.model}_random_init.keras')
