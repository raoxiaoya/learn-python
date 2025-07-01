'''
conda activate tensorflow2.19.0

人像美肤
https://modelscope.cn/models/iic/cv_unet_skin-retouching/summary

'''
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

skin_retouching = pipeline(Tasks.skin_retouching,model='iic/cv_unet_skin-retouching')

img_path = 'skin_retouching_examples_1.jpg'
dst_path = 'skin_retouching_examples_1_beauty.jpg'

# https://modelscope.oss-cn-beijing.aliyuncs.com/demo/skin-retouching/skin_retouching_examples_1.jpg

result = skin_retouching(img_path)
cv2.imwrite(dst_path, result[OutputKeys.OUTPUT_IMG])
print('finished!')

img1 = cv2.imread(img_path, cv2.IMREAD_COLOR)
img2 = cv2.imread(dst_path, cv2.IMREAD_COLOR)
cv2.imshow(img_path, img1)
cv2.imshow(dst_path, img2)
cv2.waitKey(0)
