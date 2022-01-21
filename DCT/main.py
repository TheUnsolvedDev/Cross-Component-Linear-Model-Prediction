import cv2
import numpy as np
import tensorflow as tf
import os
import sys


def quantize(qName):
    Q10 = np.array([[80, 60, 50, 80, 120, 200, 255, 255],
                    [55, 60, 70, 95, 130, 255, 255, 255],
                    [70, 65, 80, 120, 200, 255, 255, 255],
                    [70, 85, 110, 145, 255, 255, 255, 255],
                    [90, 110, 185, 255, 255, 255, 255, 255],
                    [120, 175, 255, 255, 255, 255, 255, 255],
                    [245, 255, 255, 255, 255, 255, 255, 255],
                    [255, 255, 255, 255, 255, 255, 255, 255]])

    Q50 = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 130, 99]])

    Q90 = np.array([[3, 2, 2, 3, 5, 8, 10, 12],
                    [2, 2, 3, 4, 5, 12, 12, 11],
                    [3, 3, 3, 5, 8, 11, 14, 11],
                    [3, 3, 4, 6, 10, 17, 16, 12],
                    [4, 4, 7, 11, 14, 22, 21, 15],
                    [5, 7, 11, 13, 16, 12, 23, 18],
                    [10, 13, 16, 17, 21, 24, 24, 21],
                    [14, 18, 19, 20, 22, 20, 20, 20]])
    if qName == "Q10":
        return Q10
    elif qName == "Q50":
        return Q50
    elif qName == "Q90":
        return Q90
    else:
        return np.ones((8, 8))


def imdct(block):
    return cv2.dct(block)


def imidct(block):
    return cv2.idct(block)

def imquantize(block,name = "Q50"):
	q = quantize(name)
	return block / q

def iminverse_quantize(block,name = "Q50"):
	q = quantize(name)
	return block * q


def perform_dct(image):
    dct_image = np.zeros(image.shape)
    for i in range(0, image.shape[0], 8):
        for j in range(0, image.shape[1], 8):
            dct_image[i:i+8, j:j+8] = imdct(image[i:i+8, j:j+8])
    return dct_image


def perform_idct(image):
    idct_image = np.zeros(image.shape)
    for i in range(0, image.shape[0], 8):
        for j in range(0, image.shape[1], 8):
            idct_image[i:i+8, j:j+8] = imidct(image[i:i+8, j:j+8])
    return idct_image


if __name__ == '__main__':
    image_cw = cv2.imread('./lenna.png', 1)
    image = cv2.imread('./lenna.png', 0)

    yuv = cv2.cvtColor(image_cw, cv2.COLOR_BGR2YUV)
    uv = yuv[:, :, 1:]

    res = image - 128.0
    res_dct = perform_dct(res)
    cv2.imwrite('dct_lenna.png', res_dct)

    res_idct = perform_idct(res_dct)
    image = res_idct + 128
    cv2.imwrite('idct_lenna.png', image)
