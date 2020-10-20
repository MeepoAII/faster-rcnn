import os
import xml.etree.ElementTree as ET
import numpy as np
# from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
from dataset import preprocess

from util import read_image
from util import resize_bbox
import torch


class Dataset(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        filename_list = os.listdir(os.path.join(data_dir, "img"))
        self.ids = [file.split(".")[0] for file in filename_list]


    def __getitem__(self, index):
        # 1. get image
        img_path = os.path.join(self.data_dir, "img", self.ids[index]+".jpg")
        f = Image.open(img_path)
        img = np.asarray(f, dtype=np.float32)
        # (H, W, C) -> (C, H, W)
        img = img.transpose((2, 0, 1))

        # 2. get bbox, top left and bottom right (y_min, x_min, y_max, x_max)
        xml_path = os.path.join(self.data_dir, "ano", self.ids[index]+".xml")
        anno = ET.parse(xml_path)
        bbox = list()
        label = list()

        for obj in anno.findall("object"):
            bbox_ano = obj.find("bndbox")
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bbox_ano.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')
            ])
            label.append(1)
        
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.float32)

        # resize image and bbox according paper
        # scale image
        _, H, W = img.shape
        img = preprocess(img)
        _, o_H, o_W = img.shape


        # resize bbox
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))



        return img, bbox, label



    # test if do not implement len method
    # def __len__(self):
    #     return len(self.img_list)


def test_img_read():
    test = Dataset(data_dir="/home/507/myfiles/code/meepo/faster-rcnn/mydata")
    img, bbox, label = test.__getitem__(0)
    img = img.transpose((1, 2, 0))
    # print(img)
    print(img.shape)
    plt.imshow(img.astype(np.uint8))
    plt.show()
    print(bbox)
    print(label)

test_img_read()
