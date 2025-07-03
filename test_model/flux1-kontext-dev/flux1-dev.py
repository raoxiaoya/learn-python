'''
文档
https://docs.comfy.org/zh-CN/tutorials/flux/flux-1-kontext-dev
https://mp.weixin.qq.com/s/GJ3vQUeZeiSCJ1DJdR_fzA
https://mp.weixin.qq.com/s/4ivr_tA9OaHljRfjMhRIDg
https://mp.weixin.qq.com/s/9mmaddEVs7KYTHtoHNBcjg

官方平台：playground.bfl.ai，可以在线体验

fp8和fp16代表浮点数精度floating point，8位和16位，量化版本

运行需要20G显存左右，GPU

ComfyUI适配的模型

Diffusion Model
https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors

VAE Model
https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/blob/main/split_files/vae/ae.safetensors

Text Encoder 两个
https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors
https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors 
或 
https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors


Comfy-Org 与 comfyanonymous 有什么区别

Black Forest Libs（黑森林实验室） 的模型
https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev

目前 Flux.1 Kontext 只能用英文提示词。


在 ConfyUI 中，点击左侧菜单的 “加载模板工作流”，在侧边栏找到 “Flux”，你就能看到官方预设好的两个 Kontext 工作流。这两个工作流，一个叫 Basic，一个叫 Grouped。Basic 工作流：基础的工作流，支持单图和双图输入。Grouped 工作流：这个版本更简洁，有一个大的Image Edit节点，让你的工作区清爽无比。

'''

'''
名称：FLUX.1 Kontext
开源免费，性能超过openai的GPT-4o
FLUX.1 Kontext 采用了一种名为流匹配变换器 (Flow Matching Transformer) 的全新架构。

FLUX.1 Kontext 是 Black Forest Labs 推出的突破性多模态图像编辑模型，支持文本和图像同时输入，能够智能理解图像上下文并执行精确编辑。其开发版是一个拥有 120 亿参数的开源扩散变压器模型，具有出色的上下文理解能力和角色一致性保持，即使经过多次迭代编辑，也能确保人物特征、构图布局等关键元素保持稳定。

与 FLUX.1 Kontext 套件具备相同的核心能力： 角色一致性：在多个场景和环境中保留图像的独特元素，例如图片中的参考角色或物体。 局部编辑：对图像中的特定元素进行有针对性的修改，而不影响其他部分。 风格参考：根据文本提示，在保留参考图像独特风格的同时生成新颖场景。 交互速度：图像生成和编辑的延迟极小。

虽然之前发布的 API 版本提供了最高的保真度和速度，但 FLUX.1 Kontext [Dev] 完全在本地机器上运行，为希望进行实验的开发者、研究人员和高级用户提供了无与伦比的灵活性。

版本说明
[FLUX.1 Kontext [pro] - 商业版本，专注快速迭代编辑
FLUX.1 Kontext [max] - 实验版本，更强的提示遵循能力
FLUX.1 Kontext [dev] - 开源版本（本教程使用），12B参数，主要用于研究

目前在 ComfyUI 中，你可以使用所有的这些版本，其中 Pro 及 Max 版本 可以通过 API 节点来进行调用，而 Dev 版本开源版本请参考本篇指南中的说明。

'''