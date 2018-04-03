# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

import zmq
import multiprocessing as mp
from config import config
from utils.dpflow.serialize import loads, dumps
import dataset

def data_sender(id, name, *args):
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.connect('ipc://@{}'.format(name))

    print('start data provider {}-{}'.format(name, id))
    while True:
        data_iter = dataset.train_dataset(id + 1)
        for msg in data_iter:
            # print(id)
            sender.send(dumps([id, msg]))


def provider(nr_proc, name, *args):
    proc_ids = [i for i in range(nr_proc)]

    procs = []
    for i in range(nr_proc):
        w = mp.Process(
            target=data_sender,
            args=(proc_ids[i], name, *args))
        w.deamon = True
        procs.append(w)

    for p in procs:
        p.start()


# , dataset.train_dataset()

def receiver(name):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind('ipc://@{}'.format(name))

    while True:
        id, msg = loads(receiver.recv())
        # print(id, end='')
        yield msg


if __name__ == "__main__":
    from IPython import embed
    import time
    provider(config.nr_dpflows, config.program_name)
    dataiter = receiver(config.program_name)
    start = time.clock()
    time.sleep(10)
    for i in range(1000):
        hehe = next(dataiter)
    end = time.clock()
    print("read: %f s" % (end - start))