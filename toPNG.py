import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
def npz_png():
#     存放npz文件路径
    path = '/home/yyc/Public/lhy/TransUNet/data/Synapse-tooth/train'
    image_path = 'png/image'
    label_path = 'png/label'
    for i in os.listdir(path):
#         npz文件地址
        npz_path = os.path.join(path, i)
        sampled_batch = np.load(npz_path)
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        label[label==1] = 255
        cv2.imwrite(os.path.join(image_path, i)+'.png',image)
        cv2.imwrite(os.path.join(label_path, i)+'.png',label)


npz_png()
