from procession import *
import numpy as np
import sys
import cv2


# Function to process the DCT 2D src_matrix to result_matrix.
def dct2d(im: np.ndarray):
    return mathsTool.dct_calc(src_matrix = im, is_dct = False)

# Function to process the inverse DCT 2D src_matrix to result_matrix.
def idct2d(im: np.ndarray):
    return mathsTool.dct_calc(src_matrix = im, is_dct = False)

# Function to process the imcompression.
def imcompress(coef: np.ndarray, r: float):
    my_image = image_processor()
    return my_image.imcompress_process(coef, r, True)

# Function to process the imdecompression.
def imdecomp(coef: np.ndarray): 
    my_image = image_processor()
    result_image = my_image.imcompress_process(coef, 1, False)
    for y in range(result_image.shape[1]):
        for x in range(result_image.shape[0]):
            result_image[y][x] = int(np.round(np.real(result_image[y][x])))
    return result_image

# the main function.
# how to use: python main.py sample.png 3
# means that input 
def main(name: str, rate: float):
    try:
        print(f"The input file is {name} and the rate is ", str(rate))
        my_image = image_processor()
        my_image.process_image(name, float(rate))
    except Exception as err:
        print(
            "Guide: try python main.py [filename] [rate]\n",
            "An example could be :\n\n",
            "python main.py \"sample.png\" 3\n\n",
            "Exception: ", err
        )
try:
    main(sys.argv[1], sys.argv[2])
except Exception as err:
    print(
        "Guide: try python main.py [filename] [rate]\n",
        "An example could be :\n\n",
        "python main.py \"sample.png\" 3\n\n",
        "Exception: ", err
    )
