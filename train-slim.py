import tflearn
import h5py
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import regression
from tflearn.metrics import R2
import tensorflow as tf
import tensorflow.contrib.slim as slim



model_version = "12"
model_run = "1"
model_file = '../Models/'+model_version+'/'+model_run+'/model'
image_dimensions = (240, 320)

tr_f = h5py.File("../Data/train-"+str(image_dimensions[0])+"x"+str(image_dimensions[1])+".hdf5", 'r')
X = tr_f['X']
Y = tr_f['Y']

val_f = h5py.File("../Data/val-"+str(image_dimensions[0])+"x"+str(image_dimensions[1])+".hdf5", 'r')
X_val = val_f['X']
Y_val = val_f['Y']

te_f = h5py.File("../Data/test-"+str(image_dimensions[0])+"x"+str(image_dimensions[1])+".hdf5", 'r')
X_test = te_f['X']
Y_test = te_f['Y']

chunk_size = 5120
batch_size = 128
val_batch_size = 128
test_batch_size = 128
num_batches = int(len(Y)/chunk_size)
num_epochs = 5

with tf.device('/cpu:0'):
  tflearn.config.init_training_mode()
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.9, soft_placement=True)

with tf.device('/gpu:0'):
    x = tflearn.input_data(shape=[None, 224, 224, 3], name='input')

    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    conv = fully_connected(x, 2, restore=False)

    conv = regression(conv, optimizer='adam', metric=R2(),
                         loss=tf.losses.mean_squared_error,
                         learning_rate=0.001, restore=False)
    model = tflearn.DNN(conv, tensorboard_dir=str("../Models/" + model_version + '/' + model_run + '/Tensorboard/'),
                        tensorboard_verbose=1)
    '''model.fit(X, Y, n_epoch=1, shuffle=True, batch_size=64,
                validation_set=(X_val[0:val_batch_size], Y_val[0:val_batch_size]),
                validation_batch_size=64, show_metric=True, run_id=str("Model-"+model_version+'-'+model_run),
              snapshot_step=50)
'''
model.load(model_file='../vgg_16.ckpt')
val_set_length = len(Y_val)
total_chunks_done = 0
for j in range(num_epochs):
    for i in range(num_batches):
        start = chunk_size*i
        stop = start+chunk_size
        val_start = np.random.randint(0, val_set_length-val_batch_size-1)
        val_stop = val_start+val_batch_size
        with tf.device('/GPU:0'):
            model.fit(X[start:stop], Y[start:stop], n_epoch=1, shuffle=True, batch_size=batch_size,
                      validation_set=(X_val[val_start:val_stop], Y_val[val_start:val_stop]),
                      validation_batch_size=val_batch_size, show_metric=True,
                      run_id=str("Model-"+model_version+'-'+model_run), snapshot_step=50, snapshot_epoch=False)
        rand_example = np.random.randint(0, val_set_length)
        pred = model.predict([X_val[rand_example]])
        print("Model Prediction:" + str(pred[0]))
        print("Actual Value:" + str(Y_val[rand_example]))
        print("Pixel Distance:" + str(np.sqrt(np.sum(np.square(pred[0]-Y_val[rand_example])))))
        print("Num Chunks Completed: "+str(total_chunks_done))
        if total_chunks_done%5 ==0:
            model.save(model_file=str(model_file+str(j)+str(i)))
            num_under_five = 0
            test_ds_len = len(Y_test)
            '''for t in range(test_ds_len):
                pred = model.predict([X_test[t]])
                dist = np.sqrt(np.sum(np.square(np.subtract(pred[0], Y_test[t]))))
                if dist <= 5:
                    num_under_five += 1
            print("Model Testing Accuracy: " + str(num_under_five/test_ds_len))'''
        total_chunks_done += 1

tr_f.close()
val_f.close()
