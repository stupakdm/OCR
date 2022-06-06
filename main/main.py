import glob
import os
from multiprocessing.dummy import Pool as ThreadPool
from modify.image_processing import Passport


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
    print(path)
    pas1 = Passport(path[0])
    pas1.test_quality2(0, path[0])
    change_density(path[1] + 1, path[0])

if __name__ == '__main__':
    correct_files = [2,3, 4,7,9]
    #for i in correct_files:
    #    filenames.append(f'templates/orig{i}.jpeg')

    filenames = [(f'templates/orig{i}.jpeg', i) for i in correct_files]

    os.environ['OMP_THREAD_LIMIT'] = '16'
    pool_size = len(correct_files)
    #poos = ThreadPool(pool_size)
    with ThreadPool(pool_size) as p:
        p.map(running_config, filenames)    #iap
    #list(map(running_config, filenames))

    #Using Passport
    #os.environ['OMP_THREAD_LIMIT'] = '4'
    """for (count, path) in enumerate(filenames):
        print(path)
        pas1 = Passport(path)
        pas1.test_quality2(0, path)
        change_density(count+1, path)"""