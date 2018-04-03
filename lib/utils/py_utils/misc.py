# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

import os


def ensure_dir(path):
    """create directories if *path* does not exist"""
    if not os.path.isdir(path):
        os.makedirs(path)


def parse_devices(gpu_ids):
    if '-' in gpu_ids:
        gpus = gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        parsed_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
        return parsed_ids
    else:
        return gpu_ids


if __name__ == '__main__':
    gpu_ids = "0-7"
    print(parse_devices(gpu_ids))
    gpu_ids = "0,1,2,3,4,5,6,7"
    print(parse_devices(gpu_ids))
