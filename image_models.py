import tensorflow as tf
from tensorflow.keras import layers,models

def build_vgg16(input_shape):
    # Random initialization is intentional: weights are not loaded from ImageNet.
    base=tf.keras.applications.VGG16(include_top=False,weights=None,input_shape=input_shape)
    x=layers.Flatten()(base.output); x=layers.Dense(256,activation='relu')(x); out=layers.Dense(1,activation='sigmoid')(x)
    return models.Model(base.input,out,name='VGG16_random_init')

def build_resnet50(input_shape):
    base=tf.keras.applications.ResNet50(include_top=False,weights=None,input_shape=input_shape)
    x=layers.GlobalAveragePooling2D()(base.output); x=layers.Dense(256,activation='relu')(x); out=layers.Dense(1,activation='sigmoid')(x)
    return models.Model(base.input,out,name='ResNet50_random_init')

def compile_image_model(model,lr=.001):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr,beta_1=.9,beta_2=.999,epsilon=1e-8),loss='binary_crossentropy',metrics=['accuracy'])
    return model
