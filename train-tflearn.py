import tflearn
import h5py
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import regression
import tensorflow as tf

model_version = "1"
model_run = "0"
model_file = '../Models/'+model_version+'/'+model_run+'/model.tflearn'

tr_f = h5py.File("../Data/train.hdf5", 'r')
X = tr_f['X']
Y = tr_f['Y']

val_f = h5py.File("../Data/test.hdf5", 'r')
X_val = val_f['X']
Y_val = val_f['Y']

conv = input_data(shape=[None, 120, 160, 3], dtype=tf.float32)
with tf.device('/GPU:0'):
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = max_pool_2d(conv, 3, strides=2)
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = max_pool_2d(conv, 3, strides=2)
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = fully_connected(conv, 2)
conv = regression(conv, optimizer='adam', metric='R2',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)
model = tflearn.DNN(conv, tensorboard_dir=str("../Models/"+model_version+'/'+model_run+'/Tensorboard/'),
                    tensorboard_verbose=1)
#model.load('../Models/0/0/model.tflearn')
model.fit(X, Y, n_epoch=1, shuffle=True, batch_size=256, validation_set=(X_val, Y_val), validation_batch_size=256,
          show_metric=True, run_id=str("Model-"+model_version+'-'+model_run), snapshot_step=50)
model.save(model_file=model_file)

tr_f.close()
val_f.close()