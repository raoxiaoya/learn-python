from PIL import Image
from pix2tex.cli import LatexOCR

'''
pip install pix2tex
pip install "pix2tex[gui]"
pip install -U typing_extensions
'''

img = Image.open('../imgs/360截图20231128085054222.jpg')
model = LatexOCR()
print(model(img))

# 打印信息
# \Gamma(z)=\int_{0}^{\infty}t^{z-1}e^{-t}d t\,.
