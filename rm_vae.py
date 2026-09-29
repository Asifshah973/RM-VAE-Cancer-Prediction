import tensorflow as tf
from tensorflow.keras import layers, Model

class RMVAE(Model):
    def __init__(self,input_dim,latent_dim,hidden_units=(512,256),decoder_units=None,**kwargs):
        super().__init__(**kwargs)
        decoder_units=decoder_units or tuple(reversed(hidden_units))
        self.latent_dim=latent_dim
        self.encoder_layers=[layers.Dense(int(u),activation='relu',name=f'enc_dense_{i}') for i,u in enumerate(hidden_units,1)]
        self.mu_layer=layers.Dense(latent_dim,name='mu')
        self.logvar_layer=layers.Dense(latent_dim,name='logvar')
        self.decoder_layers=[layers.Dense(int(u),activation='relu',name=f'dec_dense_{i}') for i,u in enumerate(decoder_units,1)]
        self.out_layer=layers.Dense(input_dim,activation=None,name='reconstruction')
    def encode(self,x,training=False):
        h=x
        for l in self.encoder_layers: h=l(h,training=training)
        return self.mu_layer(h),self.logvar_layer(h)
    def reparameterize(self,mu,logvar):
        eps=tf.random.normal(tf.shape(mu)); return mu+tf.exp(0.5*logvar)*eps
    def decode(self,z,training=False):
        h=z
        for l in self.decoder_layers: h=l(h,training=training)
        return self.out_layer(h,training=training)
    def call(self,x,training=False):
        mu,logvar=self.encode(x,training); z=self.reparameterize(mu,logvar); rec=self.decode(z,training)
        return rec,mu,logvar,z

def kl_divergence(mu,logvar):
    return -0.5*tf.reduce_mean(tf.reduce_sum(1+logvar-tf.square(mu)-tf.exp(logvar),axis=1))
