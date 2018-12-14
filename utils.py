


import os
import struct
import numpy as np
import os
import cv2 as cv
from scipy.misc import imresize
import random
import math
import pickle


def load_mnist(path,kind='train'):
    labels_path=os.path.join(path,'%s-labels-idx1-ubyte'%kind)
    images_path=os.path.join(path,'%s-images-idx3-ubyte'%kind)
    with open(labels_path,'rb') as lbpath:
        magic,n=struct.unpack('>II',lbpath.read(8))
        labels=np.fromfile(lbpath,dtype=np.uint8)

    with open(images_path,'rb') as imgpath:
        magic,num,rows,cols=struct.unpack('IIII',imgpath.read(16))
        images=np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels),28,28)

    return images,labels


def load_data():
    xtrain, ytrain = load_mnist("../dataset/minist/",kind='train')
    xtest, ytest = load_mnist("../dataset/minist", kind='t10k')

    return (xtrain,ytrain), (xtest,ytest)


def celebA_iter(path = "../dataset/celeba_train/"):
    img_list = os.listdir(path)
    random.shuffle(img_list)
    for i in range(int(len(img_list)/128)):
        j = 0
        images = []
        while j<128:
            img = cv.imread(path + img_list[i*128+j])
            if(img is None):
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            images.append((img.astype(np.float32) - 127.5)/127.5)
            j = j+1
        yield np.array(images)

def signal_xHz(A, fi, time_s, sample):
    return A * np.sin(np.linspace(0, fi * time_s * 2 * np.pi , sample* time_s))

def sin():
    for i in range(1000):
        train_batchs = []
        j=0
        while(j<64):
            fi = random.randint(0, 10)
            A = random.random()
            #print (fi, A)
            x = signal_xHz(A, fi, 1, 25)
            #x = np.arange(0,  2*np.pi,  2*np.pi/25)
            #x = np.sin(x)
            if np.max(x) == 0 :
                continue
            train_batchs.append(x.reshape(25,1))
            j = j+1
        yield np.array(train_batchs)


def load_abnormal(path = "../dataset/abnormal/"):
    for i in range(1000):
        ab_list = os.listdir(path)
        random.shuffle(ab_list)
        re = []
        for j in range(64):
            fp = os.path.join(path, ab_list[j])
            with open(fp, 'rb') as f:
                x = np.array(pickle.load(f)).reshape(25, 1)
                x[np.where(x > 12)] = 12
            re.append(x)
        yield np.array(re)/6 - 1
    """
    for i in range(int(len(img_list)/128)):
        j = 0
        images = []
        while j<128:
            img = cv.imread(path + img_list[i*128+j])
            if(img is None):
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            images.append((img.astype(np.float32) - 127.5)/127.5)
            j = j+1
        yield np.array(images)
    """
