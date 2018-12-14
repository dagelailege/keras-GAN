import os
import cv2 as cv
from scipy.misc import imresize
import random

# root path depends on your computer
root = './img_align_celeba/'
save_root = './celeba_train/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
print len(img_list)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = cv.imread(root + img_list[i])
    img = imresize(img, (resize_size, resize_size,3))
    cv.imwrite(filename = save_root + img_list[i], img = img)
    if (i % 5000) == 0:
        print('%d images complete' % i)
