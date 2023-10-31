# NUS-4268-Image-Procession
This is a simple guide about how to process image procession by using DCT, DFT and a bunch of other signal conversion methods.

## Introduction

The so-called DCT Transform (Discrete Cosine Transform) is a transform related to the Fourier
transform, which is similar to the DFT for Discrete Fourier Transform, but uses only real numbers. 
Discrete Fourier transform needs complex operation, although FFT can improve the operation
speed, it is very inconvenient in image coding, especially in real-time processing. Discrete Fourier
transform is rarely used in actual image communication systems, but it has theoretical significance.
According to the property of the discrete Fourier transform, the Fourier transform of the real dual
function contains only the real cosine term, so a kind of transformation of the real number domain,
the discrete cosine transform (DCT), is constructed. DCT is regarded as one of the basic processing
modules in a series of international standards for video compression coding published in recent years.

The target of the project is to use the discrete cosine transform and the inverse discrete cosine
transform to process a compression of images. For image procession, we need to introduce the 2D
DCT since the image could be considered as a matrix where the row dimension represents the width
and the column dimension represents the height. An image is composed of pixels, and the colour of
each pixel can be represented by a specific numerical value. When we think about colour, it could
be represented by a number representing the colour direction in three different dimensions: red,
green, and blue. When considering black-and-white images, the value is taken in a range between
0 (black) and 255 (white).

## 2D DCT
The formula of the Discrete Cosine Transform is as follows:
<img width="668" alt="image" src="https://github.com/CuiyuanLiu/NUS-4268-Image-Procession/assets/34060865/1daf3400-57d2-40f2-afd4-c4044bc12154">

Its inverse transform has the formula:
<img width="690" alt="image" src="https://github.com/CuiyuanLiu/NUS-4268-Image-Procession/assets/34060865/a531b455-e611-45a9-aed8-432db981401c">

<img width="432" alt="image" src="https://github.com/CuiyuanLiu/NUS-4268-Image-Procession/assets/34060865/deabf5e4-37cb-4872-a845-b124fde87efc">

The steps of the DCT compression could be summarised as:
- 1. Image Dividing, this step is to partition the image into many small 8 x 8 blocks for subsequent
processing, each taking 64 pixels.
- 2. DCT Procession, this step is to perform a discrete cosine transform for each small block. From
the formula introduced in the second part, the formula can be concretely used with N = 8,
M = 8 for a 8 × 8 block. A compression with rate r could also be processed in this step, by
picking the $\frac{64}{r}$ biggest blocks (in magnitude) and then could approach choosing the biggest $\frac{MN}{r}$ pixels.
- 3. iDCT, this step is to convert the DCT calculated results back to integer measuring the colour.
Then we could return or generate a corresponding processed image.

The logic could be:
<img width="859" alt="image" src="https://github.com/CuiyuanLiu/NUS-4268-Image-Procession/assets/34060865/6d7ddd52-c766-4964-8125-125dd160ad08">

A simple guide to use this project is to run the python file `main.py` by inputting `python main.py
[filename] [rate]` in the command prompt, a sample could be,
`python main.py ”sample.png” 3.7` and the result could be seen in the result folder with the latest timestamp.

Below is a simple example of it:
Original, size: 38kb

![sample](https://github.com/CuiyuanLiu/NUS-4268-Image-Procession/assets/34060865/1a0ed111-8f46-4c4e-b1b9-fdddbdd1c06f)

Compressed with rate 100, size: 21kb

![0_color_image](https://github.com/CuiyuanLiu/NUS-4268-Image-Procession/assets/34060865/4bb8ab68-c0e7-4f77-a001-63e4da08408e)


## Conclusion
The main idea of using DCT for compression is actually relatively simple, the image is transformed
by DCT, and then by setting a threshold then could filter the image in each block, and then use the
inverse DCT transform to transform the array back into the image. In this way, the compression
based on DCT transformation is completed.

## Reference
1. DCT变换与图像压缩、去燥 Zhou Xuhui, https://zhaoxuhui.top/blog/2018/05/26/DCTforImageDenoising.html#3dct%E5%8F%8D%E5%8F%98%E6%8D%A2idct
2. The Discrete Cosine Transform (DCT), Dave Marshall, https://users.cs.cf.ac.uk/dave/Multimedia/node231.html


