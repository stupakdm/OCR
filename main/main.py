import glob
import os, time
from multiprocessing.dummy import Pool as ThreadPool
from modify.image_processing import Passport
import concurrent.futures
from multiprocessing import Process
import asyncio

def change_density(count, path):
    if count ==0:
        os.system('mkdir temp1/')
    os.system('cp ' + path + ' temp1/')
    file_list = glob.glob('temp1/*')
    for i in file_list:
        if 'orig' in i:
            os.system('mv ' + i + ' temp1/test.jpeg')
            break
    os.system('mogrify -set density 300 temp1/test.jpeg')
    pas1 = Passport(path)
    pas1.full_process_ocr('temp1/test.jpeg')
    #pas1.test_quality2(count, path)
    print(file_list)
    os.system('rm temp1/test.jpeg')
    return None

def running_config(path):
    #t1 =time.time()
    print(path)
    pas1 = Passport(path[0])
    #await asyncio.sleep(0)
    pas1.test_quality2(0, path[0])
    #await asyncio.sleep(0)
    t1 = time.time()
    change_density(path[1] + 1, path[0])
    t2 = time.time()
    delta_time = t2-t1
    print('time used for ', delta_time, '\n')
    file1 = open(f'text+{path[1]}.txt', 'w')
    file1.write(str(delta_time))
    file1.close()
    #input()
    #await asyncio.sleep(0)

if __name__ == '__main__':

    # Файлы для тестирования модели
    correct_files = [11, 12, 13, 14,15,16]#[9, 11, 12, 13]#[2, 3, 4, 7, 9, 13, 14, 15,16]#[1,2,3,4,7,9][3, 4, 7, 9] #[1,2,3,4,7,8,9]#[1, 2, 3, 4, 7, 9] #, 2, 3, 4, 7,9]
    #correct_files = [2,3,4,7,9]
    #for i in correct_files:
    #    filenames.append(f'templates/orig{i}.jpeg')

    filenames = [(f'templates/orig{i}.jpeg', i) for i in correct_files]

    os.environ['OMP_THREAD_LIMIT'] = '4'
    pool_size = len(correct_files)
    #poos = ThreadPool(pool_size)

    # Parallel with asyncio
    #loop = asyncio.get_event_loop()
    #tasks = [loop.create_task(running_config(file)) for file in filenames]
    #wait_tasks = asyncio.wait(tasks)
    #loop.run_until_complete(wait_tasks)
    #loop.close()

    # Parallel with ThreadPool
    #with ThreadPool(pool_size) as p:
    #    p.map(running_config, filenames)    #iap

    #Parallel with Process
    """procs = []
    for i in filenames:
        try:
            p = Process(target=running_config, args=(i,))
            procs.append(p)
            p.start()
        except:
            p.join()
            procs.remove(p)

    [proc.join() for proc in procs]"""

    # Parallel with concurent.futures
    #with concurrent.futures.ThreadPoolExecutor(max_workers=len(correct_files)) as executor:
    #    executor.map(running_config, filenames)

    list(map(running_config, filenames))

    #Using Passport
    #os.environ['OMP_THREAD_LIMIT'] = '4'
    """for (count, path) in enumerate(filenames):
        print(path)
        pas1 = Passport(path)
        pas1.test_quality2(0, path)
        change_density(count+1, path)"""