import glob
import os, time
import cv2
import matplotlib.pyplot as plt
from Updating.get_all_info import Diplom
from multiprocessing.dummy import Pool as ThreadPool
import concurrent.futures
from multiprocessing import Process
import asyncio

if __name__ == '__main__':
    diplom_names = [8,9]#[3,4,5,6,7]   #1,2
    paths = []
    for i in diplom_names:
        paths.append(f'templates/diplom{i}.jpg')
    for path in paths:
        t1 = time.time()
        diplom = Diplom(path)
        diplom.conveer()
        image = cv2.imread(path)
        t2 = time.time()
        print("time for work:", t2-t1)
        #plt.imshow(image)
        #plt.show()

