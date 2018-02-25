import tflearn
import h5py
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import regression
import tensorflow as tf

model_version = "12"
model_run = "1"
epoch = "5"
batch = "5"

model_file = '../Models/'+model_version+'/'+model_run+'/model'+epoch+batch
image_dimensions = (240, 320)

te_f = h5py.File("../Data/test-"+str(image_dimensions[0])+"x"+str(image_dimensions[1])+".hdf5", 'r')
X = te_f['X']
Y = te_f['Y']
with tf.device('/cpu:0'):
    conv = input_data(shape=[None, image_dimensions[0], image_dimensions[1], 1], dtype=tf.float32)
    conv = conv-tf.reduce_min(conv)
    input_conv = -1*((conv/tf.reduce_max(conv))-0.5)

    conv = conv_2d(input_conv, 4, 9, activation='leaky_relu')
    conv = conv_2d(conv, 4, 9, activation='leaky_relu')
    conv = max_pool_2d(conv, 2, 2)
    conv = conv_2d(conv, 8, 9, activation='leaky_relu')
    conv = max_pool_2d(conv, 2, 2)
    conv = conv_2d(conv, 8, 7, activation='leaky_relu')
    conv = max_pool_2d(conv, 2, 2)
    conv = conv_2d(conv, 8, 7, activation='leaky_relu')
    conv = conv_2d(conv, 8, 7, activation='leaky_relu')
    conv = max_pool_2d(conv, 2, 2)
    conv = conv_2d(conv, 8, 5, activation='leaky_relu')
    conv = conv_2d(conv, 8, 5, activation='leaky_relu')
    conv = conv_2d(conv, 8, 5, activation='leaky_relu')
    conv = flatten(conv)
    conv = fully_connected(conv, 256, activation='leaky_relu')
    conv = fully_connected(conv, 2)
    conv = regression(conv, optimizer='adam',
                         loss=tf.losses.mean_squared_error,
                         learning_rate=0.001)
    model = tflearn.DNN(conv)
model.load(model_file)
#eval = model.evaluate(X, Y, batch_size=256)
#print(eval)

'''
LOGIC - Get model to first evaluate to a good R^2 value and then you can evaluate the under 5-pixel error, etc using:
'''

num_under_five = 0
num_under_ten = 0
num_under_fifteen = 0
test_ds_len = len(Y)
for i in range(len(Y)):
    pred = model.predict([X[i]])
    dist = np.sqrt(np.sum(np.square(np.subtract(pred[0], Y[i])), axis=1, keepdims=True))
    if dist <= 5:
        num_under_five += 1
    if dist <= 10:
        num_under_ten += 1
    if dist <= 15:
        num_under_fifteen += 1
print("Model Testing 5 Pixels Accuracy: " + str(num_under_five/test_ds_len))
print("Model Testing 10 Pixels Accuracy: " + str(num_under_ten/test_ds_len))
print("Model Testing 15 Pixels Accuracy: " + str(num_under_fifteen/test_ds_len))

te_f.close()