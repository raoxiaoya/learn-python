'''
conda activate tensorflow2.19.0
'''

import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

img_cartoon = pipeline(Tasks.image_portrait_stylization, 
                       model='iic/cv_unet_person-image-cartoon_compound-models')
# 图像本地路径
img_path = '20250630105841.png'
# 图像url链接
# img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_cartoon.png'
result = img_cartoon(img_path)
cv2.imwrite('20250630105841'+'_carton.png', result[OutputKeys.OUTPUT_IMG])
print('finished!')
