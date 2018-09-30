# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""

from config import config
import argparse
import os, sys
from pyadastools.adas import ADAS
from pyadastools.adaseval import ADASeval
from IPython import embed

def adasval(detected_json):
    eval_json = config.eval_json
    eval_gt = ADAS(eval_json)

    eval_dt = eval_gt.loadRes(detected_json)
    adasEval = ADASeval(eval_gt, eval_dt, iouType='bbox')

    # adasEval.params.imgIds = eval_gt.getImgIds()
    adasEval.evaluate()
    adasEval.accumulate()
    adasEval.summarize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str, help='json to eval')

    args = parser.parse_args()
    adasval(args.json)
    # from config import config

    # eval_json = config.eval_json
