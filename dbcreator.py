# DBCreator.py
# Created By Shreyas Hukkeri

import cv2
import numpy as np
import tensorflow as tf

def write_to_tfrecords(data, writer):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    for i in range(len(data)):
        image = data[i][0].tobytes()
        xlabel = data[i][1][0]
        ylabel = data[i][1][1]
        example = tf.train.Example(features=tf.train.Features(feature={
            'X': _bytes_feature(image),
            'Yx': _int64_feature(xlabel),
            'Yy': _int64_feature(ylabel)
        }))
        writer.write(example.SerializeToString())


def get_data_from_file(filename, image_dimensions):
    """
    A function that accepts the filename of the case and
    returns the accompanying frames and coordinate labels
    :param filename: filename with location of the file to be read
    :return: a 2-d array of video frames and their corresponding coordinate labels
    """
    def preprocess_image(data, image_dimensions):
        image = cv2.resize(data, (image_dimensions[1], image_dimensions[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        return image
    cap = cv2.VideoCapture(filename+'.avi')
    file = open(filename+'.txt')
    list_labels = file.readlines()
    frame_number = 0
    temp_array = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = preprocess_image(frame, image_dimensions)
            text_labels = list_labels[frame_number].strip().split(' ')
            arr_labels = np.asarray([int(float(text_labels[0])), int(float(text_labels[1]))])
            temp_array.append([frame, arr_labels])
            frame_number += 1
        else:
            break
    cap.release()
    return temp_array

def main(labels_file, data_directory, image_dimensions):
    file = open(labels_file)
    file_data = file.readlines()
    del file_data[0]
    tr_writer = tf.python_io.TFRecordWriter(data_directory + 'train-' + str(image_dimensions[0]) + "x" +
                                         str(image_dimensions[1]) +".tfrecords")
    te_writer = tf.python_io.TFRecordWriter(data_directory + 'test-' + str(image_dimensions[0]) + "x" +
                                         str(image_dimensions[1]) +".tfrecords")
    tv_writer = tf.python_io.TFRecordWriter(data_directory + 'val-' + str(image_dimensions[0]) + "x" +
                                         str(image_dimensions[1]) +".tfrecords")
    for line in file_data:
        line_data = line.split('\t')
        #print(line_data)
        participant = line_data[0]
        part = line_data[1]
        part_path = data_directory+'LPW/'+participant+'/'+part
        print(part_path)
        data_to_store = get_data_from_file(part_path, image_dimensions)
        num_train = int(0.8*len(data_to_store))
        num_val = int(0.05*len(data_to_store))
        num_test = len(data_to_store)-num_val-num_train
        write_to_tfrecords(data_to_store[:num_train], tr_writer)
        write_to_tfrecords(data_to_store[num_train:num_val+num_train], tv_writer)
        write_to_tfrecords(data_to_store[-num_test:], te_writer)
    tr_writer.close()
    te_writer.close()
    tv_writer.close()


if __name__ == "__main__":
    image_dimensions = (120, 160)
    data_directory = "../Data/"
    labels_file = "../Data/LPW/labels.txt"
    #get_data_from_file('../Dataset/LPW/1/1')
    main(labels_file, data_directory, image_dimensions)