import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import itertools
import math
import skimage.measure    
from scipy.stats import entropy

class LZW:
    def __init__(self):

        # self.height = image.shape[0]
        # self.width = image.shape[1]
        self.dictionary = {}
        self.dictionary_size = 256
        self.compressed = []
        self.decompressed = []
        self.entropy = 0
        self.comp = 0
        self.block_size = 8

    def compress_image(self, image, block_size, codes_size):
        # print(self.isGrayscale(image))

        self.image = image

        

        if not(self.isGrayscale(image)):
            image = np.asarray(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = np.asarray(image)

        nparray = np.asarray(image)
        

        new_width = math.ceil(self.image.size[0]/self.block_size)*self.block_size
        new_height = math.ceil(self.image.size[1]/self.block_size)*self.block_size

        new_height = new_height - self.image.size[1]
        new_width = new_width - self.image.size[0]
        
        image = np.pad(nparray, ((0,new_height),(0,new_width)), mode='constant') 
        

        self.shape = image.shape

        if block_size != -1:
            self.block_size = block_size
            blocks = self.split_into_blocks(image, block_size)
        else:
            self.block_size = block_size
            blocks = [self.nparray_to_list(image)]
            # blocks = [image]

        # print(blocks)

        # print(blocks[160])

        compressed_blocks = []
        for block in blocks:
            # print("hey")
            compressed_block = self.compress_block(block)
            compressed_blocks.append(compressed_block)
        
        # print(compressed_blocks)
        self.encoded_image = compressed_blocks
        self.encoded_copy = compressed_blocks

        max = 0
        for block in compressed_blocks:
            for element in block:
                if element > max:
                    max = element

        if max > 2**codes_size:
            assert False, "Codes size is too small"
            return

        self.write_encoded_data('compressed.txt')

        return compressed_blocks, max
        # print(blocks)

    def decompress_image(self):

        blocks = self.encoded_image

        # print(blocks[160])
        
        decompressed_blocks = []


        for block in blocks:
            # print(block)
            decompressed_block = self.decompress(block)
            refractor = []
            for element in decompressed_block:
                temp = element.split('-')
                for i in temp:
                    refractor.append(i)
            decompressed_blocks.append(refractor)
            # decompressed_blocks.append(decompressed_block)
            # break

        # print(decompressed_blocks[160])

        # print(blocks[0])

        # refractor = []
        # for block in decompressed_blocks:
        #     for element in block:
        #         temp = element.split('-')
        #         for i in temp:
        #             refractor.append(i)

        # print(refractor)
        # decompressed_blocks = refractor

        new_img = np.zeros((self.shape[0], self.shape[1]))

        # print(len(decompressed_blocks))
        # print(self.block_size)

        ct = 0

        if self.block_size != -1:

            for i in range(0, self.shape[0], self.block_size):  
                for j in range(0, self.shape[1], self.block_size):
                        # print(i, j) # 40, 0
                        temp_img = np.zeros((self.block_size, self.block_size))
                        for k in range(0, self.block_size):
                            for l in range(0, self.block_size):
                                # print(k, l, temp_img.shape, ct, self.block_size)
                                temp_img[k][l] = decompressed_blocks[ct][k*self.block_size + l]
                        new_img[i:i+self.block_size, j:j+self.block_size] = temp_img
                        # new_img[i:i+self.block_size, j:j+self.block_size] = decompressed_blocks[ct]
                        ct+=1
            #             tiles.append(tile)
            #             c += 1
                    
        else:
            # print(decompressed_blocks)
            new_img = decompressed_blocks[0]
            new_img = np.reshape(new_img, (self.shape[0], self.shape[1]))
            # print(new_img.shape)
        
        # print(new_img)
        # print(np.int32(self.YUV2RGB(new_img)))
        # self.decoded_image = np.int32(self.YUV2RGB(new_img))
        self.decoded_image = np.int32(new_img)
        self.copy_decoded = self.decoded_image.copy()
        return self.decoded_image
        
        return decompressed_blocks
    
    def decompress_image_from_file(self, file):
        """
        This function will read the compressed file and decompress it
        """
        with open(file, 'r') as f:
            lines = f.readlines()
            # print(lines[0])
            info = lines[0].split(' ')
            block_size = int(info[2])
            height = int(info[0])
            width = int(info[1])
            # print(block_size, height, width)

            self.shape = (height, width)
            self.block_size = block_size

            # print(lines[1].strip().split(','))

            # for k in lines[1].strip().split(','):
                
                # print(k.replace("[", "").replace(" ", "").replace("]", ""))

            compressed_blocks = []
            for i in range(1, len(lines)):
                block = []
                for elem in lines[i].strip().split(','):
                    temp = elem.replace("[", "").replace(" ", "").replace("]", "")
                    block.append(int(temp))
                compressed_blocks.append(block)
            
            self.encoded_image = compressed_blocks

            # print(compressed_blocks)

            return self.decompress_image()
    
    def nparray_to_list(self, array):
        final_array = []
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                final_array.append(str(array[i][j]))

        return final_array

    def isGrayscale(self, image):
        if(image.mode == 'L'):
            return True
        elif image.mode=='RGB':
            return False

    def compress_block(self, uncompressed):
        

        dict_size = 256
        dictionary = dict((str(i), str(i)) for i in range(dict_size))

        w = ""
        result = []
        for c in uncompressed:
            # wc = w + c
            if w == "":
                wc = c
            else:
                wc = c + '-' + w
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                dict_size += 1
                w = c

        if w:
            result.append(dictionary[w])

        final = []
        for element in result:
            final.append(int(element))
        return final
    
    def split_into_blocks(self, image, block_size):

        tiles = []

        for i in range(0, image.shape[0], block_size):  
            for j in range(0, image.shape[1], block_size):

                    tile = image[i:i+block_size, j:j+block_size]
                    tiles.append(self.nparray_to_list(tile))

        return tiles
    
    def decompress(self, compressed):
        """Decompress a list of output ks to a string."""

        # Build the dictionary.
        dict_size = 256
        dictionary = dict((i, str(i)) for i in range(dict_size))

        res = []
        w = str(compressed.pop(0))
        res.append(w)
        for k in compressed:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                entry = w + '-' + w.split('-')[0]
            else:
                raise ValueError('Bad compressed k: %s' % k)
            res.append(entry)            
            
            dictionary[dict_size] = w + '-' + entry.split('-')[0]
            dict_size += 1

            w = entry
        return res
    
    def write_encoded_data(self, file):
        """Write encoded data to text file"""
        with open(file, 'w') as f:
            f.write(str(self.shape[0]) + ' ' + str(self.shape[1]) + ' ' + str(self.block_size) + '\n')
            for i in range(len(self.encoded_image)):
                f.write(str(self.encoded_image[i]) + '\n')

    def getCompressionRatio(self):
        input = self.shape[0] * self.shape[1] * 8
        l = 0
        for i in range(len(self.encoded_copy)):
            l += len(self.encoded_copy[i])
        output = l*8
        # print(input, output)
        return input/output
    
    def getEntropy(self):
        N = 256
        hist, _ = np.histogram(self.image, N, range=(0, N-1))
        p_dist = hist /np.sum(hist)
        en = entropy(p_dist, base=2)
        
        return en
    
        # N = 256
        # hist, _ = np.histogram(self.image.ravel(), N, range=(0,N-1))
        # p_dist = hist/np.sum(hist)
        # en = entropy(p_dist,)
        # # print(en)
        # return en
    def get_avg_length_encoded_pixels(self):
        l = 0
        for i in range(len(self.encoded_copy)):
            l += len(self.encoded_copy[i])
        return l/len(self.encoded_copy)
    
    def get_total_no_of_codes(self):
        l = 0
        # print(self.encoded_copy[0])
        for i in range(len(self.encoded_copy)):

            l += len(self.encoded_copy[i])
        return l