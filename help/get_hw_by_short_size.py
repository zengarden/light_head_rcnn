import numpy as np

def get_hw_by_short_size(im_height, im_width, short_size, max_size):
    im_size_min = np.min([im_height, im_width]) #im_xx这里是图片的宽/高
    im_size_max = np.max([im_height, im_width]) 
    
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max

    resized_height, resized_width = int(round(im_height * scale)), int(
        round(im_width * scale))
    return resized_height, resized_width

if __name__ == "__main__":
    a,b = get_hw_by_short_size(10,10,4,9)
    print(a,b)
