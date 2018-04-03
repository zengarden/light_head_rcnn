# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

import threading

class PrefetchingIter:
    '''
    iters:  DataIter, must have forward to get  
    '''

    def __init__(self, iters, num_gpu):
        self.iters = iters
        self.n_iter = 1
        self.data_ready = [threading.Event() for _ in range(self.n_iter)]
        self.data_taken = [threading.Event() for _ in range(self.n_iter)]
        for e in self.data_taken:
            e.set()
        self.started = True
        self.current_batch = [None for _ in range(self.n_iter)]
        self.next_batch = [None for _ in range(self.n_iter)]

        def prefetch_func(self, i):
            """Thread entry"""
            while True:
                self.data_taken[i].wait()
                if not self.started:
                    break
                try:
                    blobs_list = []
                    cnt = 0
                    while cnt < num_gpu:
                        blobs = next(iters)
                        if blobs['is_valid']:
                            cnt += 1
                            blobs_list.append(blobs)

                    #for gpu_id in range(num_gpu):
                    #    blobs = next(iters)
                    #    blobs_list.append(blobs)
                    self.next_batch[i] = blobs_list
                except StopIteration:
                    self.next_batch[i] = None
                self.data_taken[i].clear()
                self.data_ready[i].set()

        self.prefetch_threads = [
            threading.Thread(target=prefetch_func, args=[self, i]) \
            for i in range(self.n_iter)]
        for thread in self.prefetch_threads:
            thread.setDaemon(True)
            thread.start()

    def __del__(self):
        self.started = False
        for e in self.data_taken:
            e.set()
        for thread in self.prefetch_threads:
            thread.join()

    def iter_next(self):
        for e in self.data_ready:
            e.wait()
        if self.next_batch[0] is None:
            return False
        else:
            self.current_batch = self.next_batch[0]
            for e in self.data_ready:
                e.clear()
            for e in self.data_taken:
                e.set()
            return True

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        if self.iter_next():
            return self.current_batch
        else:
            raise StopIteration
