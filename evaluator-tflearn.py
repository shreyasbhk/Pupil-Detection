import tflearn
import h5py
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import regression
import tensorflow as tf

model_version = "3"
model_run = "1"
model_file = '../Models/'+model_version+'/'+model_run+'/model'
image_dimensions = (180, 240)

te_f = h5py.File("../Data/test-"+str(image_dimensions[0])+"x"+str(image_dimensions[1])+".hdf5", 'r')
X = te_f['X']
Y = te_f['Y']

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
conv = regression(conv, optimizer='adam',
                     loss='mean_square',
                     learning_rate=0.001)
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