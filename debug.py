import numpy as np
import torch.multiprocessing as mp
import time
import copy


class MultiProcess():
    def __init__(self, pf):
        self.worker_nums = 5
        self.manager = mp.Manager()
        self.train_epochs = 5
        self.pf = copy.deepcopy(pf)
        self.shared_func = copy.deepcopy(self.funcs)
        self.Time = 0
        self.start_worker()
        
    @staticmethod
    def f(name, Time, shared_pf, shared_que, start_barrier, epochs, start_epoch):
        
        pf = copy.deepcopy(shared_pf)
        current_epoch = 0
        
        while True:
            start_barrier.wait()
            current_epoch += 1
            pf = shared_pf
            if current_epoch < start_epoch:
                shared_que.put({
                    'rst': None,
                })
                continue
            if current_epoch > epochs:
                break
            shared_que.put({
                'rst': name,
                'time': Time,
                'pf': pf
            })
            time.sleep(1)
    
    @property
    def funcs(self):
        return {
            "pf": self.pf
        }
    
    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)
        
        start_epoch = 0
        for name in ['alice','bob','carol','dave','eve']:
            p = mp.Process(target=self.__class__.f, args=(name, self.Time, self.shared_func['pf'], self.shared_que, self.start_barrier, self.train_epochs, start_epoch))
            p.start()
            self.workers.append(p)
    
    def train_one_epoch(self):
        # self.shared_func = copy.deepcopy(self.pf)
        # print(self.pf)
        active_worker_nums = 0
        result = []
        for _ in range(self.worker_nums):
            worker_rst = self.shared_que.get()
            result.append(worker_rst['rst'])
            print(worker_rst)
        self.active_worker_nums = active_worker_nums



if __name__ == '__main__':
    pf = 0
    collector = MultiProcess(pf)
    for _ in range(5):
        collector.train_one_epoch()
        collector.pf += 1
    

        
    

