import numpy as np, tensorflow as tf
from src.models.dnn import build_dnn

class DNNTrainer:
    def __init__(self,input_dim,hidden_units=(128,64),dropout=0,lr=.001,batch_size=32,epochs=100,patience=10,seed=1234):
        tf.keras.utils.set_random_seed(seed)
        self.model=build_dnn(input_dim,hidden_units,dropout)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr,beta_1=.9,beta_2=.999,epsilon=1e-8),loss='binary_crossentropy',metrics=['accuracy'])
        self.batch_size=batch_size; self.epochs=epochs; self.patience=patience
    def fit(self,X,y,X_val=None,y_val=None):
        callbacks=[]
        if X_val is not None:
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=self.patience,restore_best_weights=True)]
            val=(X_val,y_val)
        else: val=None
        self.history=self.model.fit(X,y,batch_size=self.batch_size,epochs=self.epochs,validation_data=val,callbacks=callbacks,verbose=0)
        return self
    def predict_proba(self,X): return self.model.predict(X,verbose=0).ravel()
