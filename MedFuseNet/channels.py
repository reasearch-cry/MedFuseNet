import os
import numpy as np
file='/root/HiFormer_main/ACDC/test/case_003_volume_ED.npz'
t=np.load(file)
image, label = t['img'], t['label']
print(image.shape)
print(label.shape)