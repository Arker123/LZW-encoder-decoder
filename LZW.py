import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import itertools
import math

class LZW:
    def __init__(self, image):
        self.image = image
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.dictionary = {}
        self.dictionary_size = 256
        self.compressed = []
        self.decompressed = []
        self.entropy = 0
        self.comp = 0

    def compress(self, uncompressed):
        

        dict_size = 256
        dictionary = dict((str(i), i) for i in range(dict_size))

        w = ""
        result = []
        for c in uncompressed:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                dict_size += 1
                w = c

        if w:
            result.append(dictionary[w])
        return result
    
    def decompress(self, compressed):
        """Decompress a list of output ks to a string."""
        from io import StringIO

        # Build the dictionary.
        dict_size = 256
        dictionary = dict((i, str(i)) for i in range(dict_size))
        # in Python 3: dictionary = {i: chr(i) for i in range(dict_size)}
        
    #     print(dictionary)

        # use StringIO, otherwise this becomes O(N^2)
        # due to string concatenation in a loop
        res = []
        result = StringIO()
        w = str(compressed.pop(0))
        result.write(w)
        res.append(w)
    #     print(result.getvalue())
        for k in compressed:
    #         print(k)
            if k in dictionary:
    #             print("HEHE")
                entry = dictionary[k]
            elif k == dict_size:
    #             print("HERE")
    #             print(w)
                entry = w + '-' + w.split('-')[0]
            else:
                raise ValueError('Bad compressed k: %s' % k)
            result.write(entry)
            res.append(entry)

            # Add w+entry[0] to the dictionary.
            
            
    #         print("Entry: ", entry)
            dictionary[dict_size] = w + '-' + entry.split('-')[0]
            dict_size += 1

            w = entry
    #     print(dictionary)
        return res
    
