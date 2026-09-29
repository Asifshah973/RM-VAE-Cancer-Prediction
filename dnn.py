import tensorflow as tf
from tensorflow.keras import layers, Model

def build_dnn(input_dim, hidden_units=(128,64), dropout=0.0):
    x_in=layers.Input((input_dim,),name='latent_input')
    x=x_in
    for i,u in enumerate(hidden_units,1):
        x=layers.Dense(int(u),activation='relu',name=f'dnn_dense_{i}')(x)
        if dropout>0: x=layers.Dropout(dropout,name=f'dnn_dropout_{i}')(x)
    out=layers.Dense(1,activation='sigmoid',name='risk_probability')(x)
    return Model(x_in,out,name='DNN_classifier')
