import tflearn
import h5py
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import regression
from tflearn.metrics import R2
import tensorflow as tf

model_version = "1"
model_run = "1"
model_file = '../Models/'+model_version+'/'+model_run+'/model.tflearn'

tr_f = h5py.File("../Data/train.hdf5", 'r')
X = tr_f['X']
Y = tr_f['Y']

val_f = h5py.File("../Data/val.hdf5", 'r')
X_val = val_f['X']
Y_val = val_f['Y']

conv = input_data(shape=[None, 120, 160, 3], dtype=tf.float32)
with tf.device('/GPU:0'):
    conv = ((conv-tf.reduce_min(conv))/tf.reduce_max(conv))-0.5
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = max_pool_2d(conv, 3, strides=2)
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = max_pool_2d(conv, 3, strides=2)
    conv = conv_2d(conv, 16, 3, activation='leaky_relu')
    conv = fully_connected(conv*100, 2)
conv = regression(conv, optimizer='adam', metric=R2(),
                     loss='mean_square',
                     learning_rate=0.0001)
model = tflearn.DNN(conv, tensorboard_dir=str("../Models/"+model_version+'/'+model_run+'/Tensorboard/'),
                    tensorboard_verbose=1)

batch_size = 512
num_batches = int(len(Y)/256)

for i in range(num_batches):
    start = batch_size*i
    stop = start+batch_size
    model.fit(X[start:stop], Y[start:stop], n_epoch=1, shuffle=True, batch_size=256,
              validation_set=(X_val[0:int(batch_size/2)], Y_val[0:int(batch_size/2)]),
              validation_batch_size=256, show_metric=True, run_id=str("Model-"+model_version+'-'+model_run),
              snapshot_step=50)
    pred = model.predict([X[i]])
    print("Model Prediction:" + str(pred[0]))
    print("Actual Value:" + str(Y[i]))
    print("Pixel Distance:" + str(np.sqrt(np.sum(np.square(pred[0]-Y[i])))))
    model.save(model_file=model_file)

tr_f.close()
val_f.close()