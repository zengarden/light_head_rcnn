import os,sys,re
import numpy as np

'''
class_list = ['car', 'ped']
label_map = {
    # cars, buses, other vehicles
    'p': 0,
    'f': 0,
    'w': 0,
    # pedestrians, rider, occulude pedestrians
    'q': 1,
    's': 1,
    'o': 1,
}

class_list = ['ped']
label_map = {
    # pedestrians, rider, occulude pedestrians
    'q': 0,
    's': 0,
    'o': 0,
}

'''

def convert(size, box):
    # input: x1,x2,y1,y2
    # out:   x,y,w,h
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def invconvert(size, box):
    x = (box[0]-box[2]/2.) * size[0]
    y = (box[1]-box[3]/2.) * size[1]
    w = box[2]*size[0]
    h = box[3]*size[1]
    ret = (x+1,y+1,w,h)
    #print('box is ', box, 'size is ', size, 'ret is ', ret)
    ret2 = [int(i+0.5) for i in ret]
    return ret2


def image2label(imgfn, darknet_style = True):
    """
    txtfn = imgfn
    txtfn = re.sub(r'\.(jpg|png|bmp|jpeg)$', r'.txt', txtfn, re.I)
    if darknet_style:
        txtfn = re.sub(r'images/', r'labels/', txtfn)
        txtfn = re.sub(r'JPEGImages/', r'labels/', txtfn)
        txtfn = re.sub(r'raw/', r'labels/', txtfn)
    if txtfn == imgfn:
        print("warning: the image path %s is not right!" % imgfn )
    return txtfn
    """

def recursive_get_images(data_dir, pattern = r'\.jpg$'):
    rootdir = os.path.abspath(data_dir)
    for root, dirs, files in os.walk(rootdir):
        #print(root, dirs, files)
        for f in files:
            if re.search(pattern, f, re.I):
                absfn = os.path.join(root, f)
                #print('new file %s' % absfn)
                yield absfn

# load data
# if catid is None, the load all the data
def load_data(fn, catid = None, thresh = None):
    data = {}
    nbox = 0
    for line in open(fn).readlines():
        fields = line.strip().split(' ')
        image_id = fields[0]

        if image_id not in data:
            data[image_id] = {
                'bbox': [],
                'dummy': []
            }
 
        if len(fields) == 1:
            continue

        if catid != None:
            if int(fields[1]) != catid:
                continue
                
        if thresh != None:
            if float(fields[1]) < thresh:
                continue

        data[image_id]['bbox'].append(np.array([float(x) for x in fields[2:6]]))
        data[image_id]['dummy'].append(float(fields[1]))
        nbox += 1
    print("%s has %d images, %s boxes" % (fn, len(data), str(nbox)))
    return (data, nbox)


if __name__ == "__main__":
    image_paths = recursive_get_images("/home/zuosi/data-detnet/valid.800.v3")
    for img in image_paths:
        print(img)
