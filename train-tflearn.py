from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()
import tflearn
import h5py
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import regression
from tflearn.metrics import R2
import tensorflow as tf


model_version = "4"
model_run = "1"
model_file = '../Models/'+model_version+'/'+model_run+'/model'
image_dimensions = (240, 320)
train_dataset_file = "../Data/train-"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"
val_dataset_file = "../Data/val-"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"
test_dataset_file = "../Data/test-"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"

num_epochs = 5
batch_size = 32
val_batch_size = 64


def initialize_datasets():
    with tf.device("/cpu:0"):
        def parser_function(example_proto):
            features = {
                "X": tf.FixedLenFeature((), tf.string, default_value=""),
                "Yx": tf.FixedLenFeature((), tf.int64),
                "Yy": tf.FixedLenFeature((), tf.int64)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            image = tf.reshape(tf.decode_raw(parsed_features["X"], tf.float32),
                               [image_dimensions[0], image_dimensions[1], 1])
            xlabel = tf.reshape(tf.cast(parsed_features["Yx"], tf.float32), [1])
            ylabel = tf.reshape(tf.cast(parsed_features["Yy"], tf.float32), [1])
            label = tf.concat([xlabel, ylabel], axis= 0)
            return image, label
        dataset = tf.data.TFRecordDataset(train_dataset_file)
        dataset = dataset.map(parser_function)
        dataset = dataset.repeat(1)
        dataset = dataset.shuffle(batch_size*3)
        dataset = dataset.batch(batch_size)

        val_dataset = tf.data.TFRecordDataset(val_dataset_file)
        val_dataset = val_dataset.map(parser_function)
        val_dataset = val_dataset.repeat(1)
        val_dataset = val_dataset.batch(val_batch_size)

        test_dataset = tf.data.TFRecordDataset(test_dataset_file)
        test_dataset = test_dataset.map(parser_function)
        test_dataset = test_dataset.repeat(1)
        test_dataset = test_dataset.batch(val_batch_size)
        return dataset, val_dataset, test_dataset

with tf.device('/GPU:0'):
    conv = input_data(shape=[None, 240, 320, 3], dtype=tf.float32)
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
model = tflearn.DNN(conv, tensorboard_dir=str("../Models/"+model_version+'/'+model_run+'/Tensorboard/'),
                    tensorboard_verbose=1)

with tf.device("/GPU:0"):
    for j in range(num_epochs):
        tr_ds, v_ds, t_ds = initialize_datasets()
        for (batch, (X, Y)) in enumerate(tfe.Iterator(tr_ds)):
            model.fit(X, Y, n_epoch=1, shuffle=True, batch_size=batch_size, show_metric=True,
                      run_id=str("Model-"+model_version+'-'+model_run), snapshot_step=50)
            pred = model.predict([X[1]])
            print("Model Prediction:" + str(pred[0]))
            print("Actual Value:" + str(Y[1]))
            print("Pixel Distance:" + str(np.sqrt(np.sum(np.square(pred[0]-Y[1])))))
            if batch%20 == 0:
                model.save(model_file=str(model_file+str(j)+str(batch)))
