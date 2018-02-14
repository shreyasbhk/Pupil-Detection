import h5py
import cv2
import numpy as np

def initialize_datasets(location, image_dimensions):
    f = h5py.File(location+'/train.hdf5', 'w')
    dsetx = f.create_dataset("X", (1, image_dimensions[0], image_dimensions[1], 3), maxshape=(None, image_dimensions[0], image_dimensions[1], 3))
    dsety = f.create_dataset("Y", (1, 2), maxshape=(None, 2))
    f.close()
    f = h5py.File(location+'/val.hdf5', 'w')
    dsetx = f.create_dataset("X", (1, image_dimensions[0], image_dimensions[1], 3), maxshape=(None, image_dimensions[0], image_dimensions[1], 3))
    dsety = f.create_dataset("Y", (1, 2), maxshape=(None, 2))
    f.close()
    f = h5py.File(location+'/test.hdf5', 'w')
    dsetx = f.create_dataset("X", (1, image_dimensions[0], image_dimensions[1], 3), maxshape=(None, image_dimensions[0], image_dimensions[1], 3))
    dsety = f.create_dataset("Y", (1, 2), maxshape=(None, 2))
    f.close()

def write_to_hdf5(data, dataset_file):
    f = h5py.File(dataset_file, 'a')
    dsetx = f['X']
    dsety = f['Y']
    num_examples = len(data)
    old_len = len(dsety)
    dsetx.resize(old_len+num_examples, axis=0)
    dsety.resize(old_len+num_examples, axis=0)
    for i in range(num_examples):
        dsetx[old_len+i] = data[i][0]
        dsety[old_len+i] = data[i][1]
    f.close()


def get_data_from_file(filename, image_dimensions):
    """
    A function that accepts the filename of the case and
    returns the accompanying frames and coordinate labels
    :param filename: filename with location of the file to be read
    :return: a 2-d array of video frames and their corresponding coordinate labels
    """
    def preprocess_image(data, image_dimensions):
        image = cv2.resize(data, (image_dimensions[1], image_dimensions[0]))
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
            arr_labels = [np.float32(text_labels[0]), np.float32(text_labels[1])]
            temp_array.append([frame, arr_labels])
            frame_number += 1
        else:
            break
    cap.release()
    return temp_array

def main(labels_file, data_directory, image_dimensions):
    initialize_datasets(data_directory, image_dimensions)
    file = open(labels_file)
    file_data = file.readlines()
    del file_data[0]
    for line in file_data:
        line_data = line.split('\t')
        participant = line_data[0]
        part = line_data[1]
        part_path = data_directory + 'LPW/' + participant + '/' + part
        print(part_path)
        data_to_store = get_data_from_file(part_path, image_dimensions)
        num_train = int(0.8 * len(data_to_store))
        num_val = int(0.05 * len(data_to_store))
        num_test = len(data_to_store) - num_val - num_train
        write_to_hdf5(data_to_store[:num_train], data_directory+"/train.hdf5")
        write_to_hdf5(data_to_store[num_train:num_val + num_train], data_directory+"/val.hdf5")
        write_to_hdf5(data_to_store[-num_test:], data_directory+"/test.hdf5")

if __name__ == "__main__":
    image_dimensions = (120, 160)
    data_directory = "../Data/"
    labels_file = "../Data/LPW/labels.txt"
    #get_data_from_file('../Dataset/LPW/1/1')
    main(labels_file, data_directory, image_dimensions)