import cv2
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import regression
from tflearn.metrics import R2

model_file = '../Models/22/1/model455'
image_dimensions = (240, 320)
conv = input_data(shape=[None, image_dimensions[0], image_dimensions[1], 1], dtype=tf.float32)
conv = conv-tf.reduce_min(conv)
input_conv = -1*((conv/tf.reduce_max(conv))-0.5)
conv = conv_2d(input_conv, 4, 9, activation='leaky_relu')
conv = conv_2d(conv, 4, 9, activation='leaky_relu')
conv = max_pool_2d(conv, 2, 2)
conv = conv_2d(conv, 8, 9, activation='leaky_relu')
conv = max_pool_2d(conv, 2, 2)
conv = conv_2d(conv, 8, 7, activation='leaky_relu')
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
conv = regression(conv, optimizer='adam', metric=R2(),
                     loss=tf.losses.mean_squared_error,
                     learning_rate=0.001)
model = tflearn.DNN(conv)
model.load(model_file)


def get_camera_image(cap):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


if __name__ == "__main__":
    capt = cv2.VideoCapture(0)
    while True:
        image = get_camera_image(capt)
        cv2.imshow('frame', image)
        cv2.waitKey(1)
