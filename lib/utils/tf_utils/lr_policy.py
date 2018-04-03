# encoding: utf-8
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""
from abc import ABCMeta, abstractmethod


class BaseLR():
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter): pass


class MultiStageLR(BaseLR):
    def __init__(self, lr_stages):
        assert type(lr_stages) in [list, tuple] and len(lr_stages[0]) == 2, \
            'lr_stages must be list or tuple, with [iters, lr] format'
        self._lr_stagess = lr_stages

    def get_lr(self, epoch):
        for it_lr in self._lr_stagess:
            if epoch < it_lr[0]:
                return it_lr[1]


class LinearIncreaseLR(BaseLR):
    def __init__(self, start_lr, end_lr, warm_iters):
        self._start_lr = start_lr
        self._end_lr = end_lr
        self._warm_iters = warm_iters
        self._delta_lr = (end_lr - start_lr) / warm_iters

    def get_lr(self, cur_epoch):
        return self._start_lr + cur_epoch * self._delta_lr


if __name__ == '__main__':
    lr = LinearIncreaseLR(0.00001, 0.1, 5)
    from IPython import embed;

    embed()
    print(lr._delta_lr)
    for i in range(5):
        print(lr.get_lr(i))
