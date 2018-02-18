import tflearn
import h5py
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import regression
import tensorflow as tf

model_version = "1"
model_run = "0"
model_file = '../Models/'+model_version+'/'+model_run+'/model.tflearn'

te_f = h5py.File("../Data/test.hdf5", 'r')
X = te_f['X']
Y = te_f['Y']

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
    conv = fully_connected(conv, 2)
conv = regression(conv, optimizer='adam', metric='R2',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)
model = tflearn.DNN(conv)
model.load(model_file)
#eval = model.evaluate(X, Y, batch_size=256)
#print(eval)

'''
LOGIC - Get model to first evaluate to a good R^2 value and then you can evaluate the under 5-pixel error, etc using:
'''

num_under_five = 0

for i in range(len(Y)):
    pred = model.predict([X[i]])
    dist = np.sqrt(np.sum(np.square(pred[0]-Y[i])))
    print(pred-Y[i])
    if(dist < 5):
        num_under_five += 1
print(num_under_five/(len(Y)))

te_f.close()