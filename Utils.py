import numpy as np
import struct
import matplotlib.pyplot as plt

class Data_util(object):
    """
    load image, label
    """

    def __init__(self, filename=None):
        self._filename = filename

        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte

    def get_image(self):
        """
        load image
        :return: numpy
        """
        binfile = open(self._filename, 'rb')  # open with binary
        buf = binfile.read()
        binfile.close()
        index = 0
        numMagic, numImgs, numRows, numCols = struct.unpack_from(self._fourBytes2, buf, index)
        index += struct.calcsize(self._fourBytes)
        images = []
        for i in range(numImgs):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            for j in range(len(imgVal)):
                if imgVal[j] > 1:
                    imgVal[j] = 1
            images.append(imgVal)
        return np.array(images)

    def get_label(self):
        """
        load label
        :return: numpy
        """
        binFile = open(self._filename, 'rb')
        buf = binFile.read()
        binFile.close()
        index = 0
        magic, numItems = struct.unpack_from(self._twoBytes2, buf, index)
        index += struct.calcsize(self._twoBytes2)
        labels = []
        for x in range(numItems):
            im = struct.unpack_from(self._labelByte2, buf, index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)

class Function(object):
    """
    Functions for NN
    """
    def __init__(self):
        pass

    def out_img(self, image):
        """
        show image
        """
        img = np.array(image)
        img = img.reshape(28, 28)
        plt.figure()
        plt.imshow(img)

    def sigmoid(self):
        pass

    def relu(self):
        pass




class Preprocess(object):
    """
    Some functions for image pre-process
    """
    def __init__(self):
        pass

    def normalization(self, data):
        """
        Normalization
        :param data: numpy
        :return: numpy
        """
        mean = np.mean(data)
        std = np.std(data)
        data = (data - mean)/std
        print(mean, std)
        return data




