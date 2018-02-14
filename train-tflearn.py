import tflearn
import h5py
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import regression

model_version = "0"
model_run = "0"

tr_f = h5py.File("../Data/train.hdf5", 'r')
X = tr_f['X']
Y = tr_f['Y']

val_f = h5py.File("../Data/val.hdf5", 'r')
X_val = val_f['X']
Y_val = val_f['Y']

conv = input_data(shape=[None, 120, 160, 3], dtype=tf.float32)
conv = conv_2d(conv, 32, 3, activation='relu')
conv = conv_2d(conv, 32, 3, activation='relu')
conv = conv_2d(conv, 32, 3, activation='relu')
conv = conv_2d(conv, 32, 3, activation='relu')
conv = fully_connected(conv, 2)
conv = regression(conv, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(conv, tensorboard_dir='../Models/'+model_version+'/'+model_run+'/Tensorboard', tensorboard_verbose=0)
model.fit(X, Y, n_epoch=10, shuffle=True, batch_size=128, validation_set=(X_val, Y_val), validation_batch_size=128, show_metric=True, run_id=str("Model-"+model_version+'-'+model_run))
model.save(model_file='../Models/'+model_version+'/'+model_run+'/model.tflearn')

tr_f.close()
val_f.close()