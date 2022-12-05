import glob

import os, time
#from multiprocessing.dummy import Pool as ThreadPool
from modify_passport.get_all_info_ps import Passport
#import concurrent.futures
#from multiprocessing import Process
#import asyncio

def change_density(count, path, pasport):
    '''
    Инициализация всех начальных переменных,
    запуск конвеера
    '''
    if count ==0:
        os.system('mkdir temp1/')
    os.system('cp ' + path + ' temp1/')
    file_list = glob.glob('temp1/*')
    for i in file_list:
        if 'orig' in i:
            os.system('mv ' + i + ' temp1/test.jpeg')
            break
    os.system('mogrify -set density 300 temp1/test.jpeg')
    #pas1 = Passport(path)
    pasport.init_everything(path)
    delta_time = pasport.full_process_ocr('temp1/test.jpeg')

    #pas1.test_quality2(count, path)
    print(file_list)
    os.system('rm temp1/test.jpeg')
    return delta_time

def running_config(path, passport):
    #t1 =time.time()
    print(path)



    delta_time = change_density(path[1] + 1, path[0], passport)
    file1 = open(f'results/passport_results/time/text+{path[1]}.txt', 'w')
    file1.write(str(delta_time))
    file1.close()


    #await asyncio.sleep(0)

if __name__ == '__main__':

    # Файлы для тестирования модели
    passport_names = [ 2,3, 9, 12, 13,14, 20, 21, 22]#[ 20, 21, 22, 23, 24]#[11, 12, 13, 14,15,16]#[9, 11, 12, 13]#[2, 3, 4, 7, 9, 13, 14, 15,16]#[1,2,3,4,7,9][3, 4, 7, 9] #[1,2,3,4,7,8,9]#[1, 2, 3, 4, 7, 9] #, 2, 3, 4, 7,9]

    passport = Passport()
    filenames = [(f'media/passport_photo/orig{i}.jpeg', i) for i in passport_names]

    #os.environ['OMP_THREAD_LIMIT'] = '4'
    pool_size = len(passport_names)
    #poos = ThreadPool(pool_size)

    # Parallel with asyncio


    # Parallel with ThreadPool
    #with ThreadPool(pool_size) as p:
    #    p.map(running_config, filenames)    #iap

    #Parallel with Process


    # Parallel with concurent.futures

    for file in filenames:
        running_config(file, passport)


    #Using Passport
    #os.environ['OMP_THREAD_LIMIT'] = '4'
