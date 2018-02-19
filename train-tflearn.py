import tflearn
import h5py
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import regression
from tflearn.metrics import R2
import tensorflow as tf

model_version = "3"
model_run = "1"
model_file = '../Models/'+model_version+'/'+model_run+'/model'
image_dimensions = (180, 240)

tr_f = h5py.File("../Data/train-"+str(image_dimensions[0])+"x"+str(image_dimensions[1])+".hdf5", 'r')
X = tr_f['X']
Y = tr_f['Y']

val_f = h5py.File("../Data/val-"+str(image_dimensions[0])+"x"+str(image_dimensions[1])+".hdf5", 'r')
X_val = val_f['X']
Y_val = val_f['Y']

batch_size = 2048
val_batch_size = 128
num_batches = int(len(Y)/val_batch_size)
num_epochs = 5

with tf.device('/cpu:0'):
  tflearn.config.init_training_mode()
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.8, soft_placement=True)

with tf.device('/gpu:0'):
    conv = input_data(shape=[None, image_dimensions[0], image_dimensions[1], 1], dtype=tf.float32)
    conv = ((conv-tf.reduce_min(conv))/tf.reduce_max(conv))-0.5
    conv = conv_2d(conv, 16, 9, activation='leaky_relu')
    conv = conv_2d(conv, 16, 9, activation='leaky_relu')
    conv = conv_2d(conv, 16, 9, activation='leaky_relu')
    conv = max_pool_2d(conv, 2, strides=2)
    conv = conv_2d(conv, 16, 7, activation='leaky_relu')
    conv = conv_2d(conv, 16, 7, activation='leaky_relu')
    conv = conv_2d(conv, 16, 7, activation='leaky_relu')
    conv = max_pool_2d(conv, 2, strides=2)
    conv = conv_2d(conv, 16, 5, activation='leaky_relu')
    conv = conv_2d(conv, 16, 5, activation='leaky_relu')
    conv = max_pool_2d(conv, 2, strides=2)
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = fully_connected(conv, 2)
    conv = regression(conv, optimizer='adam', metric=R2(),
                         loss='mean_square',
                         learning_rate=0.001)
    model = tflearn.DNN(conv,tensorboard_dir=str("../Models/" + model_version + '/' + model_run + '/Tensorboard/'),
                        tensorboard_verbose=1)
    model.fit(X, Y, n_epoch=1, shuffle=True, batch_size=64,
                validation_set=(X_val[0:val_batch_size], Y_val[0:val_batch_size]),
                validation_batch_size=64, show_metric=True, run_id=str("Model-"+model_version+'-'+model_run),
              snapshot_step=50)
'''
for j in range(num_epochs):
    for i in range(num_batches):
        start = batch_size*i
        stop = start+batch_size
        with tf.device('/GPU:0'):
            model.fit(X[start:stop], Y[start:stop], n_epoch=1, shuffle=True, batch_size=64,
                      validation_set=(X_val[0:val_batch_size], Y_val[0:val_batch_size]),
                      validation_batch_size=64, show_metric=True, run_id=str("Model-"+model_version+'-'+model_run),
                      snapshot_step=50)
        pred = model.predict([X[i]])
        print("Model Prediction:" + str(pred[0]))
        print("Actual Value:" + str(Y[i]))
        print("Pixel Distance:" + str(np.sqrt(np.sum(np.square(pred[0]-Y[i])))))
        model.save(model_file=str(model_file+str(j)+str(i)))
'''
tr_f.close()
val_f.close()