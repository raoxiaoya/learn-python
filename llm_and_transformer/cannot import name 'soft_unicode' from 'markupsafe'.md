cannot import name 'soft_unicode' from 'markupsafe'



报错

```bash
Traceback (most recent call last):
  File "llm_and_transformer/bert/use_bert-base-chinese4.py", line 1, in <module>
    import gradio as gr
  File "d:\ProgramData\Anaconda3\lib\site-packages\gradio\__init__.py", line 3, in <module>
    import gradio._simple_templates
  File "d:\ProgramData\Anaconda3\lib\site-packages\gradio\_simple_templates\__init__.py", line 1, in <module>
    from .simpledropdown import SimpleDropdown
  File "d:\ProgramData\Anaconda3\lib\site-packages\gradio\_simple_templates\simpledropdown.py", line 6, in <module>
    from gradio.components.base import Component, FormComponent
  File "d:\ProgramData\Anaconda3\lib\site-packages\gradio\components\__init__.py", line 1, in <module>
    from gradio.components.annotated_image import AnnotatedImage
  File "d:\ProgramData\Anaconda3\lib\site-packages\gradio\components\annotated_image.py", line 14, in <module>
    from gradio.components.base import Component
  File "d:\ProgramData\Anaconda3\lib\site-packages\gradio\components\base.py", line 20, in <module>
    from gradio.blocks import Block, BlockContext
  File "d:\ProgramData\Anaconda3\lib\site-packages\gradio\blocks.py", line 39, in <module>
    from gradio import (
  File "d:\ProgramData\Anaconda3\lib\site-packages\gradio\networking.py", line 15, in <module>
    from gradio.routes import App  # HACK: to avoid circular import # noqa: F401
  File "d:\ProgramData\Anaconda3\lib\site-packages\gradio\routes.py", line 58, in <module>
    from fastapi.templating import Jinja2Templates
  File "d:\ProgramData\Anaconda3\lib\site-packages\fastapi\templating.py", line 1, in <module>
    from starlette.templating import Jinja2Templates as Jinja2Templates  # noqa
  File "d:\ProgramData\Anaconda3\lib\site-packages\starlette\templating.py", line 14, in <module>
    import jinja2
  File "d:\ProgramData\Anaconda3\lib\site-packages\jinja2\__init__.py", line 12, in <module>
    from .environment import Environment
  File "d:\ProgramData\Anaconda3\lib\site-packages\jinja2\environment.py", line 25, in <module>
    from .defaults import BLOCK_END_STRING
  File "d:\ProgramData\Anaconda3\lib\site-packages\jinja2\defaults.py", line 3, in <module>
    from .filters import FILTERS as DEFAULT_FILTERS  # noqa: F401
  File "d:\ProgramData\Anaconda3\lib\site-packages\jinja2\filters.py", line 13, in <module>
    from markupsafe import soft_unicode
ImportError: cannot import name 'soft_unicode' from 'markupsafe' (d:\ProgramData\Anaconda3\lib\site-packages\markupsafe\__init__.py)
```



可以看出是因为 Jinja2 引入 makeupsafe 导致的报错



查看版本

```bash
> pip show markupsafe
Name: MarkupSafe
Version: 2.1.5
Summary: Safely add untrusted strings to HTML/XML markup.
Home-page: https://palletsprojects.com/p/markupsafe/
Author:
Author-email:
License: BSD-3-Clause
Location: d:\programdata\anaconda3\lib\site-packages
Requires:
Required-by: gradio, Jinja2, Sphinx

> pip show Jinja2
Name: Jinja2
Version: 2.11.3
Summary: A very fast and expressive template engine.
Home-page: https://palletsprojects.com/p/jinja/
Author: Armin Ronacher
Author-email: armin.ronacher@active-4.com
License: BSD-3-Clause
Location: d:\programdata\anaconda3\lib\site-packages
Requires: MarkupSafe
Required-by: anaconda-project, bokeh, conda-build, conda-verify, Flask, gradio, jupyter-server, jupyterlab, jupyterlab-server, nbconvert, notebook, numpydoc, Sphinx, to
rch
```



从依赖关系上来讲 `gradio, Jinja2, Sphinx`依赖`MarkupSafe`，而`MarkupSafe`没有下游依赖。

我看到网上有人说修改`Jinja2`的版本，但是这可能导致依赖于`Jinja2`的包出现问题，所以还是修改`MarkupSafe`的版本更好，但是天知道修改到哪个版本呢。

Jinja2 仓库 https://github.com/pallets/jinja/releases?page=2

![image-20241031115209340](D:\dev\php\magook\trunk\server\md\img\image-20241031115209340.png)

makeupsafe仓库 https://github.com/pallets/markupsafe/releases

![image-20241031115306497](D:\dev\php\magook\trunk\server\md\img\image-20241031115306497.png)

这是2024年3月



所以，从时间上来看，JIaja2 的版本实在太古老。我们换成最新的 3.1.4，从时间上来看，两个包的版本能匹配上。

![image-20241031115940817](D:\dev\php\magook\trunk\server\md\img\image-20241031115940817.png)

`python -m pip install jinja2==3.1.4`



问题解决。

