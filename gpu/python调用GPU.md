python调用GPU



**环境与依赖**

机器环境，使用 modelscope notebook测试

![image-20250706150544136](D:\dev\php\magook\trunk\server\md\img\image-20250706150544136.png)

`ubuntu22.04-cuda12.1.0-py311-torch2.3.1-tf2.16.1-1.27.0`



1、安装 CUDA Toolkit
从 [NVIDIA CUDA 官网](https://developer.nvidia.com/cuda-downloads)下载并安装 CUDA Toolkit。确保选择与你的 GPU 驱动和操作系统版本兼容的版本。

2、安装 cuDNN
如果你计划进行深度学习任务，还需要安装 cuDNN。cuDNN 是 NVIDIA 提供的深度神经网络加速库，可以从 [NVIDIA cuDNN 官网](https://developer.nvidia.com/cudnn)下载。

3、安装 Python 的 CUDA 绑定库
在 Python 中，cupy 和 torch 是两个常用的库，分别用于通用计算和深度学习。

```bash
pip install cupy-cudaXX  
# 替换 XX 为你的 CUDA 版本号，例如 cupy-cuda113，并不是一一对应的，要去 https://pypi.org/search/?q=cupy-cuda 查看

pip install cupy-cuda12x

conda list cupy

# Name                    Version                   Build  Channel
cupy-cuda12x              13.4.1                   pypi_0    pypi
```

```bash
# For CUDA 11.2 ~ 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# For AMD ROCm 4.3
pip install cupy-rocm-4-3

# For AMD ROCm 5.0
pip install cupy-rocm-5-0
```



**关于 cupy**

pypi：https://pypi.org/project/cupy-cuda12x/

github：https://github.com/cupy/cupy

cupy官网:https://cupy.dev/

document：https://docs.cupy.dev/en/v13.4.1/

api reference：https://docs.cupy.dev/en/v13.4.1/reference/comparison.html



在使用方法上，可以大致将 Cupy 与 Numpy 对比，在多维数组/矩阵上进行加速，对比表 [comparison table](https://docs.cupy.dev/en/stable/reference/comparison.html)

```python
import cupy as cp

x = cp.arange(6).reshape(2, 3).astype('f')
x.sum(axis=1)
```

