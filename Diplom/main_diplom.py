import time

from Updating.get_all_info import Diplom


if __name__ == '__main__':
    diplom_names = [3,4,5,8,9]+[i for i in range(13,30)]#[13]#[3,4,5,8,9]+[i for i in range(12,21)]#[3,4,5,8,9]#[8, 10]#[3, 4, 5, 8, 9]#[3,4,5]#[3,4,5]   #1,2
    paths = []
    for i in diplom_names:
        paths.append(f'templates/diplom{i}.jpg')

    diplom = Diplom()
    for ind, path in enumerate(paths):
        d_1 = path.find('diplom') + len('diplom')
        d_2 = path.find('.')

        print('File '+str(path[d_1:d_2]))

        t1 = time.time()
        diplom.initialize(path)
        info = diplom.conveer()
        t2 = time.time()
        print("time for work:", t2 - t1)

        with open(f'Results/results/result_{path[d_1:d_2]}.txt', 'w') as f:
            for key in info.keys():
                f.write(key+' ')
                if type(info[key]) == list:
                    for item in info[key]:
                        f.write(item+', ')
                else:
                    f.write(info[key])
                f.write('\n')

        with open(f'Results/time/time_{path[d_1:d_2]}.txt', 'w') as f:
            f.write("time for work: "+str(t2-t1))


