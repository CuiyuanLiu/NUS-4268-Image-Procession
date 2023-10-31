import cv2
import numpy as np
import os
from datetime import datetime

from scipy.fft import dct, idct


class mathsTool():    
    """_summary_
    A class to store the supporting functions.
    
    Functions:mathsTool.save_image
    def innerProduct(vec_x, vec_y), Function to return the inner product between x and y.
    def dct_calc(src_matrix: np.ndarray, is_dct: bool), Function to process the dct and idct.
    def dct2d(ipt_matrix: np.ndarray), Function to process dct.
    def idct2d(ipt_matrix: np.ndarray), Function to process idct.
    def save_image(ipt_Lst: list, tag: str), Function to save the image as jpg by using np.array as input.
    """
    # Function to compute the inner product between vector x and vector y. 
    def innerProduct(vec_x, vec_y):
        result = 0        
        for i in range(np.shape(vec_x)[0]):
            for j in range(np.shape(vec_x)[0]):
                result = result + vec_x[i,j] * np.conj(vec_y[i,j])
        return result
    
    # Function to calculate the dct or idct result.
    def dct_calc(src_matrix: np.ndarray, is_dct: bool):
        M, N = np.shape(src_matrix)[0], np.shape(src_matrix)[1]
        result_matrix = np.zeros([M, N], dtype = np.complex_)
        
        for counter1 in range(0, M):
            coeff1 = np.sqrt(1/M) if (counter1 == 0) else np.sqrt(2/M)
            for counter2 in range(0, N):
                coeff2 = np.sqrt(1/N) if (counter2 == 0) else np.sqrt(2/N)
                D2_cosine = np.matmul(
                    np.cos((np.pi * (counter1) * (2 * np.arange(M) + 1)) / (2 * M)).reshape((M, 1)),
                    np.cos((np.pi * (counter2) * (2 * np.arange(N) + 1)) / (2 * N)).reshape((1, N))
                ) if (is_dct) else np.matmul(
                    np.cos((np.pi * (2 * counter1 + 1) * (np.arange(M))) / (2 * M)).reshape((M, 1)),
                    np.cos((np.pi * (2 * counter2 + 1) * (np.arange(N))) / (2 * N)).reshape((1, N))
                )
                result_matrix[counter1, counter2] = (coeff1 * coeff2) * (mathsTool.innerProduct(src_matrix, D2_cosine))
        return result_matrix
        
    # Function to process the DCT 2D src_matrix to result_matrix.
    def dct2d(ipt_matrix: np.ndarray):
        return mathsTool.dct_calc(src_matrix = ipt_matrix, is_dct = True)
    
    # Function to process the inverse DCT 2D src_matrix to result_matrix.
    def idct2d(ipt_matrix: np.ndarray):
        return mathsTool.dct_calc(src_matrix = ipt_matrix, is_dct = False)
    
    # Function to save the image under the current directory.
    def save_image(ipt_Lst: list, tag: str):
        if not os.path.exists("result"): os.mkdir("result")
        curr_time = "".join(list(str(x) for x in [
            datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute, tag
        ]))
        if not os.path.exists("result/" + curr_time): os.mkdir("result/" + curr_time)
        for item in enumerate(ipt_Lst):
            try: cv2.imwrite("/" .join(['result', curr_time, str(item[0]) + '_color_image.png']), item[1])
            except Exception as err: print("Error!", err)

class image_processor():
    """_summary_
    A class to provide an instance for the image procession.
    
    Attrubutes:
    self.image_height = N, the image vector is in N dimension. 
    self.image_width = M, R MN.
    self.rate, the ratio of compression.
    self.src_image_path, the path to store the original image.
    self.src_image, np.array read the image by using opencv.
    self.dct_image, the image processed already by using dct compression.
    self.splited_image_Lst, list to contain the blocks cut.
    
    Functions:
    def split_image(self, ipt_image, h_size: int, w_size: int), Function to cut the image into 8 * 8 blocks.
    def DCT_image(self, ipt_image: ), Function to process the DCT compression.
    def process_image(self, image_path: str, rate: float), Function to process the compression.
    
    """
    def __init__(self):
        self.image_height = 1
        self.image_width = 1
        self.rate = 1
        self.src_image_path = ""
        self.src_image = np.zeros(shape=(self.image_width, self.image_height))
        self.dct_image = self.src_image
        self.splited_image_Lst = []
    
    # Function to split the input image basing on split_size
    def split_image(self, ipt_image, h_size: int, w_size: int):
        width, height = ipt_image.shape[0], ipt_image.shape[1]
        result_list = []
        
        rows_result = (np.split(ipt_image, height // h_size))
        for curr_result in rows_result:
            counter1 = 0
            while (counter1 + 8 <= width):
                curr_unit = []
                left_num, right_num = counter1, counter1 + w_size
                for counter2 in range(0, len(curr_result)):
                    curr_unit.append(curr_result[counter2][left_num : right_num])
                counter1 = counter1 + w_size
                result_list.append(np.array(curr_unit))
        return result_list
    
    # Function to return a complex compressed block by setting a rate selecting first (MN/r) cells.
    def compress_block(self, ipt_block: np.ndarray, rate: float):
        result_block = np.zeros([ipt_block.shape[0], ipt_block.shape[1]], dtype=np.complex_)
        size = ipt_block.shape[0] * ipt_block.shape[1]
        K = int(np.round((1 / rate) * size))
        index = np.array(np.argsort(-abs(ipt_block.reshape((1, size))))[:, 0: K])[0]
        
        for counter in np.arange(len(index)):
            x = int(np.floor(index[counter] / ipt_block.shape[0]))
            y = int(index[counter] - x * ipt_block.shape[1])
            result_block[x, y] = ipt_block[x, y]    
        return result_block

    # Function to process the DCT imcompression.
    def imcompress_process(self, ipt_image: np.ndarray, rate: float, is_dct: bool):
        image_width, image_height = ipt_image.shape[0], ipt_image.shape[1]
        cut_wsize, cut_hsize = 8, 8
        splited_image_Lst = self.split_image(ipt_image, cut_wsize, cut_hsize)
        
        if (is_dct == True): result_image = np.zeros((image_width, image_height), np.complex64)
        else: result_image = np.zeros((image_width, image_height), np.uint64) 
        
        counter, counter_h = 0, 0
        while (counter_h < image_height):
            counter_w = 0
            while (counter_w < image_width):
                curr_block = splited_image_Lst[counter]
                compressed_block = mathsTool.dct_calc(curr_block, is_dct)
                if (rate != 1): compressed_block = self.compress_block(compressed_block, rate)
                result_image[counter_h : counter_h + 8, counter_w : counter_w + 8] = compressed_block
                counter_w = counter_w + 8
                counter = counter + 1
            counter_h = counter_h + 8
        return result_image
    
    # Function to process the imcompression.
    def imcompress(self, ipt_image: np.ndarray, rate: float):
        return self.imcompress_process(ipt_image, rate, True)
    
    # Function to process the imdecompression.
    def imdecomp(self, ipt_image: np.ndarray): 
        result_image = self.imcompress_process(ipt_image, 1, False)
        for y in range(result_image.shape[1]):
            for x in range(result_image.shape[0]):
                result_image[y][x] = int(np.round(np.real(result_image[y][x])))
        return result_image
        
    # Function to process the image procession.
    def process_image(self, image_path: str, rate: float):
        self.src_image_path = image_path
        self.rate = rate
        try: self.src_image = cv2.imread(self.src_image_path, cv2.IMREAD_GRAYSCALE)
        except Exception as err: print("Having issue in processing ", self.src_image_path, err)
        self.image_width  = self.src_image.shape[0]
        self.image_height = self.src_image.shape[1]
        self.dct_image = self.imcompress(self.src_image, rate)
        self.dct_image = self.imdecomp(self.dct_image)
        print(self.dct_image.shape)
        mathsTool.save_image([self.dct_image], "")
        

"""
testing part.
"""
#########################################################################################################
def test_spliting(processor: image_processor):
    splited_image_Lst = processor.split_image(processor.src_image, 128, 64)
    mathsTool.save_image(splited_image_Lst, "")

def test_cv2():
    # A comparaing version saved as name in cv2 to see effects.
    ipt_img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    opt_img = cv2.idct(cv2.dct(np.float32(ipt_img), cv2.DCT_INVERSE))
    mathsTool.save_image([opt_img], "cv2")
    
# test_spliting(my_image)
# test cv2