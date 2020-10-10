from util import random_flip, read_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread('/home/507/myfiles/data/vo'
                 'c/VOCdevkit/VOC2007/JPEGImages/007978.jpg')


img = img[::-1, :, :]
print(img.shape)

plt.imshow(img)
plt.show()