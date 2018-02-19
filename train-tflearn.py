import tflearn
import h5py
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import regression
from tflearn.metrics import R2
from tensorflow.contrib.losses import mean_pairwise_squared_error
import tensorflow as tf

model_version = "8"
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

chunk_size = 10000
batch_size = 1000
val_batch_size = 1000
test_batch_size = 1000
num_batches = int(len(Y)/chunk_size)
num_epochs = 5

with tf.device('/cpu:0'):
  tflearn.config.init_training_mode()
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.9, soft_placement=True)

with tf.device('/gpu:0'):
    conv = input_data(shape=[None, image_dimensions[0], image_dimensions[1], 1], dtype=tf.float32)
    conv = conv-tf.reduce_min(conv)
    conv = -1*((conv/tf.reduce_max(conv))-0.5)
    conv = flatten(conv)
    conv = fully_connected(conv, 1024, activation='sigmoid')
    conv = fully_connected(conv, 2)
    conv = regression(conv, optimizer='adam', metric=R2(),
                         loss=mean_pairwise_squared_error,
                         learning_rate=0.001)
    model = tflearn.DNN(conv,tensorboard_dir=str("../Models/" + model_version + '/' + model_run + '/Tensorboard/'),
                        tensorboard_verbose=1)
    '''model.fit(X, Y, n_epoch=1, shuffle=True, batch_size=64,
                validation_set=(X_val[0:val_batch_size], Y_val[0:val_batch_size]),
                validation_batch_size=64, show_metric=True, run_id=str("Model-"+model_version+'-'+model_run),
              snapshot_step=50)
'''
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
        if total_chunks_done%3 ==0:
            model.save(model_file=str(model_file+str(j)+str(i)))
            num_under_five = 0
            test_ds_len = len(Y_test)
            for t in range(int(test_ds_len/test_batch_size)-1):
                strt = test_batch_size * t
                stp = strt + test_batch_size
                #print("Start Test Num: "+str(strt))
                #print("Stop Test Num: "+str(stp))
                pred = model.predict(X_test[strt:stp])
                distances = np.sqrt(np.sum(np.square(np.subtract(pred[:], Y_test[strt:stp])), axis=1, keepdims=True))
                #print(distances)
                for dist in distances:
                    if dist <= 5:
                        num_under_five += 1
            print("Model Testing Accuracy: " + str(num_under_five/test_ds_len))
        total_chunks_done += 1

tr_f.close()
val_f.close()
