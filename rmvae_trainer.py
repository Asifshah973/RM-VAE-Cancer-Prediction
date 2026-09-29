from pathlib import Path
import numpy as np, tensorflow as tf
from src.models.rm_vae import RMVAE, kl_divergence
from src.geometry.metric import geometric_regularizer

class RMVAETrainer:
    def __init__(self,input_dim,latent_dim,hidden_units,decoder_units,beta,lam,lr,batch_size,epochs,geom_eps,proj_dim,fd_delta,seed=1234,geometry_batch_size=1):
        self.input_dim=input_dim; self.latent_dim=latent_dim; self.beta=beta; self.lam=lam
        self.batch_size=batch_size; self.epochs=epochs; self.fd_delta=fd_delta; self.geom_eps=geom_eps; self.seed=seed
        self.geometry_batch_size=geometry_batch_size
        self.model=RMVAE(input_dim,latent_dim,hidden_units,decoder_units,name='RMVAE')
        self.optimizer=tf.keras.optimizers.Adam(lr,beta_1=.9,beta_2=.999,epsilon=1e-8)
        rng=np.random.default_rng(seed)
        self.P=tf.constant(rng.normal(0,1/np.sqrt(proj_dim),size=(input_dim,proj_dim)).astype('float32'))
        self.history=[]

    def fit(self,X):
        X=np.asarray(X,np.float32)
        ds=tf.data.Dataset.from_tensor_slices(X).shuffle(len(X),seed=self.seed,reshuffle_each_iteration=True).batch(self.batch_size)
        for epoch in range(1,self.epochs+1):
            sums=np.zeros(4,float); n=0
            for xb in ds:
                with tf.GradientTape() as tape:
                    rec,mu,logvar,z=self.model(xb,training=True)
                    rec_loss=tf.reduce_mean(tf.reduce_sum(tf.square(xb-rec),axis=1))
                    kl=kl_divergence(mu,logvar)
                    # The finite-difference geometric term is evaluated on a small fixed subset
                    # to keep the exact stochastic-Jacobian formulation computationally tractable.
                    m=min(self.geometry_batch_size,int(xb.shape[0]))
                    geo=tf.add_n([geometric_regularizer(self.model.decode,z[i],self.P,self.fd_delta,self.geom_eps) for i in range(m)])/float(m)
                    loss=rec_loss+self.beta*kl+self.lam*geo
                grads=tape.gradient(loss,self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))
                b=int(xb.shape[0]); sums += np.array([float(loss),float(rec_loss),float(kl),float(geo)])*b; n+=b
            row={'epoch':epoch,'loss':sums[0]/n,'reconstruction':sums[1]/n,'kl':sums[2]/n,'geometric':sums[3]/n}
            self.history.append(row)
        return self

    def embeddings(self,X):
        mu,_=self.model.encode(tf.convert_to_tensor(X,np.float32),training=False)
        return mu.numpy()

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.model.save_weights(path)
