'''
conda activate tensorflow2.19.0

人像卡通化
https://modelscope.cn/models/iic/cv_unet_person-image-cartoon_compound-models/summary

'''

import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_cartoon = pipeline(Tasks.image_portrait_stylization, 
                       model='iic/cv_unet_person-image-cartoon_compound-models')
# 图像本地路径
img_path = '20250630105841.png'
dst_path = '20250630105841_carton.png'

# 图像url链接
# img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_cartoon.png'

result = img_cartoon(img_path)
cv2.imwrite(dst_path, result[OutputKeys.OUTPUT_IMG])
print('finished!')

img1 = cv2.imread(img_path, cv2.IMREAD_COLOR)
img2 = cv2.imread(dst_path, cv2.IMREAD_COLOR)
cv2.imshow(img_path, img1)
cv2.imshow(dst_path, img2)
cv2.waitKey(0)