# DBCreator.py
# Created By Shreyas Hukkeri

import cv2
import numpy as np

def get_data_from_file(filename):
    """

    :param filename: filename with location of the file to be read
    :return: a 2-d array of video frames and their corresponding coordinate labels
    """
    cap = cv2.VideoCapture(filename+'.avi')
    file = open(filename+'.txt')
    list_labels = file.readlines()
    frame_number = 0
    temp_array = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            text_labels = list_labels[frame_number].strip().split(' ')
            arr_labels = [np.float16(text_labels[0]), np.float16(text_labels[1])]
            temp_array.append([frame, arr_labels])
            frame_number += 1
        else:
            break
    cap.release()
    return temp_array


if __name__ == "__main__":
    get_data_from_file('../Dataset/LPW/1/1')