import cv2
import time
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
    img = cv2.resize(gray, (image_dimensions[1], image_dimensions[0]))
    return img


def get_marked_image(img):
    original_img = img
    pred_img = np.expand_dims(img.astype(np.float32), axis=-1)
    pred = model.predict([pred_img])
    print((int(pred[0][0]), int(pred[0][1])))
    color_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_img, (int(pred[0][0]/2), int(pred[0][1]/2)), 10, (0,0,255), -1)
    return color_img


def get_prediction(img):
    pred_img = np.expand_dims(img.astype(np.float32), axis=-1)
    pred = model.predict([pred_img])
    return pred


if __name__ == "__main__":
    capt = cv2.VideoCapture(1)
    capt.set(cv2.CAP_PROP_BRIGHTNESS, 3)
    start_time = time.time()
    num_frames = 0
    while True:
        image = get_camera_image(capt)
        image = get_prediction(image)
        num_frames += 1
        if(num_frames%30==0):
            last_time = time.time()
            print(num_frames/(last_time-start_time))
            num_frames=0
            start_time = last_time
        #image = get_marked_image(image)
        #cv2.imshow('frame', image)
        cv2.waitKey(1)
