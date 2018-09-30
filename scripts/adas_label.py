#!/usr/bin/env python

from __future__ import print_function
import os, sys
import re
import glob
from PIL import Image
from adas_utils import convert, image2label, recursive_get_images


def get_counters(label_map):
    counters = {}
    for i in label_map.keys():
        counters[i] = 0
    return counters


def convert_annotation(label_map, image_path, label_path, out_label_path, list_file, bnd_width, bnd_height, counters, width_stat, height_stat):
    imsize = Image.open(image_path).size
    #print('the size is' , imsize)
    in_file = open(label_path)
    if label_path == out_label_path:
        print("out_label_path may overwrite label_path, please check first!")
        sys.exit(1)
        
    out_dir = os.path.dirname(out_label_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_file = open(out_label_path, 'w')

    lines = in_file.readlines()
    
    #counters = get_counters(label_map)
    

    nbox = 0
    for line in lines:
        label = line.strip().split(',')
        key = label[0]

        if not key in label_map:
            continue

        box = [int(i) for i in label[2:6]]
        
        (x, y, width, height) = box

        if x < 0:
            box[0] = 0;
        if x+width >= imsize[0]:
            box[2] = imsize[0] - box[0];
        if y < 0:
            box[1] = 0;
        if y+height >= imsize[1]:
            box[3] = imsize[1] - box[1];

        if width >= len(width_stat):
            width_stat.extend([0] * (width - len(width_stat) + 1) )
        width_stat[width] += 1
        if height >= len(height_stat):
            height_stat.extend([0] * (height-len(height_stat) + 1) ) 
        height_stat[height] += 1

        b = (box[0], box[0] + box[2], box[1], box[1] + box[3])
        bb = convert(imsize, b)

        out_file.write(str(label_map[key]) + ' ' + ' '.join([str(a) for a in bb]) + '\n')

        counters[key]+=1
        nbox += 1
    if nbox > 0:
        list_file.write(image_path + '\n')
        pass
    return counters, width_stat, height_stat


def main(input_config):    
    # get the input config data
    _config = {}
    exec(open(input_config).read(),None, _config)
    config = type('Config', (), _config)
    #print('config is ', vars(config))

    datadict = {
        'train' : config.train_data,
        'valid' : config.valid_data,
    }
    
    for key, data in datadict.items():
        total_counter = get_counters(config.label_map)
        width_stat = []
        height_stat = []

        print('processing for category %s' % key)
        list_fn = data['list_file']

        with open(list_fn, 'w') as list_file:
            imcnt = 0
            if 'image_paths' in data:
                image_paths = data['image_paths']
            else:
                image_paths = recursive_get_images(data['image_dirs'])
            
            for image_path in image_paths:
                imcnt += 1
                if imcnt % 100 == 0:
                    #print('%d, processing image %s' % (imcnt,image_path))
                    pass
                if not os.path.exists(image_path):
                    print("no jpg file %s" % image_path)
                    continue
                label_path = image2label(image_path, False)
                if not os.path.exists(label_path):
                    print("no label file %s" % label_path)
                    continue
            
                out_label_path = image2label(image_path, True)
                total_counter, width_stat, height_stat = convert_annotation(config.label_map, image_path, label_path, out_label_path, list_file,
                                         data['bnd_width'], data['bnd_height'], total_counter, width_stat, height_stat)
            
            spl = os.path.splitext(list_fn)
            stat_fn = spl[0] + "_stat" + spl[1]
            for fn in [sys.stdout, open(stat_fn, 'w')]:
                print('total counter %s' % str(total_counter), file=fn)
                print('width_stat %s' % width_stat, file = fn)
                print('height_stat %s' % height_stat, file = fn)


    print("generate ground truth for valid data")
    import adas_gen_gt
    adas_gen_gt.main(datadict['valid']['list_file'])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_config', help ='config file input_config.py', type = str)
    args = parser.parse_args()
    main(args.input_config)
