from PIL import Image
import numpy as np

dirpath=r'D:\data\lyh\2Dimg9Se\train\images\01孙玲霞\VIBRANT+C1-JPG\P-21良-01-VIBRANT+C1-35.jpg'

img_PIL = Image.open(dirpath)#读取数据


print("img_PIL:",img_PIL)
# img_PIL: <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=2736x1856 at 0x2202A8FC108>
print("img_PIL:",type(img_PIL))
# img_PIL: <class 'PIL.JpegImagePlugin.JpegImageFile'>


#将图片转换成np.ndarray格式
img_PIL = np.array(img_PIL)
print("img_PIL:",img_PIL.shape)
# img_PIL: (1856, 2736, 3)
print("img_PIL:",type(img_PIL))
# img_PIL: <class 'numpy.ndarray