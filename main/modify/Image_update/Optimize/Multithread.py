from multiprocessing.dummy import Pool as ThreadPool
import threading


class MyThreads:
    num_of_threads = 5

    def __init__(self, num):
        self.num_of_threads = num

    def start_threading(self, func, **kwargs):
        filenames = kwargs['filenames']
        args = kwargs['args']
        sync = kwargs['sync']
        is_result = kwargs['is_result']
        results = []
        if args != None:
            pass
        with ThreadPool(self.num_of_threads) as p:
            if is_result:
                if sync:
                    results = p.imap(func, filenames)
                else:
                    results = (p.map_async(func, filenames)).get()
                return results
            else:
                return None

#MyThreads.start_threading( func, filenames='aer', args=['tgtg', 'rtgr', 0])