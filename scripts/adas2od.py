#!/usr/bin/python
# coding: utf-8

import re
import os
import pdb
import glob
import time
import json
from PIL import Image
from xml.dom.minidom import parseString
from adas_utils import recursive_get_images

G_LABEL_MAP = {
    'adas_car': {
        'ignore': ['q','s','g','o','c'],
        'multi_class': False,
        'default': 'car'
    },
    'adas_tired': {
        'ignore': [],
        'multi_class': True,
        'g': 'o',
        'q': 's'
    }
}

class Adas2OD(object):
    """
    """
    def __init__(self, dataset='adas_car', store_path="/home/zuosi/data/ADASDevkit/ADAS2017"):
        self.store_path = store_path
        self._init_label_map(dataset)

    def _init_label_map(self, dataset):
        if G_LABEL_MAP.get(dataset):
            self.label_map = G_LABEL_MAP[dataset]
        else:
            raise Exception("invalid dataset_type:{}".format(dataset))

    def parse_boxes(self, image_path):
        txt = re.sub(r'\.(jpg|jpeg|png|bmp)$', r'.txt', image_path, re.I)

        imsize = None
        try:
            imsize = Image.open(image_path).size
        except IOError as e:
            print "{0} open exception, {1}".format(image_path, e)
            return None

        boxs = []
        for l in open(txt):
            l = l.strip()
            if not l:
                continue
           
            label = l.split(',')
            if label[0] in self.label_map['ignore']:
                continue
            elif self.label_map['multi_class']:
                if label[0] in self.label_map:
                    name = self.label_map[label[0]]
                else:
                    name = label[0]
            else:
                name = self.label_map['default']

            box = [int(i) for i in label[2:6]]
            (x, y, w, h) = box
            if x < 0:
                x = 0
            if x+w >= imsize[0]:
                w = imsize[0] - x
            if y < 0:
                y = 0
            if y+h >= imsize[1]:
                h = imsize[1] - y
            
            boxs.append({
                'box': [x, y, w, h],
                'tag': name,
                'extra': {'ignore': 0}
            })

        return None if len(boxs) == 0 else boxs

    def convert(self, image_path):
        boxs = self.parse_boxes(image_path)
        if not boxs:
            print("{} no valid boxes".format(image_path))
            return None
        
        imsize = Image.open(image_path).size
        od = {
            'gtboxes': boxs,
            'fpath': image_path,
            'dbName': 'ADAS',
            'width': imsize[0],
            'height': imsize[1],
            'ID': os.path.basename(image_path),
        }
        
        return od


def run(input_config):
   # get the input config data
    _config = {}
    exec(open(input_config).read(),None, _config)
    config = type('Config', (), _config)

    datadict = config.data
  
    c = Adas2OD("adas_car", config.store_path)

    for key, data in datadict.items():
        if len(data['image_dirs']) == 0:
            continue

        odgt_path = os.path.join(
                config.store_path, 
                "odformat", 
                key+".odgt") 
        
        with open(odgt_path, 'w') as fod:
            for dir in data['image_dirs']:
                image_dir = os.path.join(config.data_path, dir)
                image_paths = recursive_get_images(image_dir)
                for image_path in image_paths:
                    gt = c.convert(image_path)
                    if not gt:
                        print("{} convert none".format(image_path))
                        continue
                    gt['fpath'] = gt['fpath'].replace(config.data_path.rstrip('/'), '')

                    fod.write(json.dumps(gt) + '\n')

 
if __name__ == "__main__":
    run("config.py")
