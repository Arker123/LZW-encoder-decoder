import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import itertools
import math
from LZW import LZW
import os

if __name__ == "__main__":

    img = Image.open('./test_images/Fig81b.tif')
    lzw = LZW()
    compressed, max = lzw.compress_image(img, 8, 16)
    lzw.write_encoded_data('compressed.txt')
    print(compressed)
    decompressed = lzw.decompress_image_from_file('compressed.txt')
    plt.imshow(decompressed, cmap='gray')
    plt.show()
        

    # #     lzw.write_encoded_data('compressed.txt')
    # #     # print(compressed)
    # #     decompressed = lzw.decompress_image_from_file('compressed.txt')
    # #     # decompressed = lzw.decompress_image()
    # #     # print(decompressed[0])
    # #     plt.imshow(decompressed, cmap='gray')
    # #     # plt.show()
    # #     print()
    # #     print(filename)
    # #     print("Max value: ", max)
    # #     print("Number of codes used: ", lzw.get_total_no_of_codes())
    # #     print("Compression Ratio: ", lzw.getCompressionRatio())
    # #     print("Average length of encoded pixels: ", lzw.get_avg_length_encoded_pixels())
    # #     print("Entropy: ", lzw.getEntropy())
    # #     print()

    # L = [4, 8, 16, 32, 64, 128]
    # CR = []

    # for i in L:
    #     img = Image.open('./test/kodim01.png')
    #     # img = Image.open('./test_images/Fig81b.tif')
    #     lzw = LZW()
    #     compressed, max = lzw.compress_image(img, i, 32)
        

    #     # lzw.write_encoded_data('compressed.txt')
    #     # print(compressed)
    #     decompressed = lzw.decompress_image_from_file('compressed.txt')
    #     CR.append(lzw.getCompressionRatio())
    #     # decompressed = lzw.decompress_image()
    #     # print(decompressed[0])
    #     # plt.imshow(decompressed, cmap='gray')
    #     # plt.show()
    #     # print()
    #     # print("Block size: ", i)
    #     # print("Max value: ", max)
    #     # print("Number of codes used: ", lzw.get_total_no_of_codes())
    #     # print("Compression Ratio: ", lzw.getCompressionRatio())
    #     # print("Average length of encoded pixels: ", lzw.get_avg_length_encoded_pixels())
    #     # print("Entropy: ", lzw.getEntropy())
    #     # print()
        
    # plt.plot(L, CR, label='Compression Ratio')
    # plt.xlabel('Block size')
    # plt.ylabel('Compression Ratio')
    # plt.title('Block size vs Compression Ratio')
    # plt.legend()
    # plt.show()
    # # l = 0
    # # for i in lzw.compress_image(img, 8, 12):
    # #     for j in i:
    # #         l += 1
    # # print(l)
    # # print(len(lzw.compress_image(img, 8, 12)))

    
    # # bw_image = np.asarray(img)
    # # print(bw_image.shape)