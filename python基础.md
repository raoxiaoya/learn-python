
#### `__pycache__`
有时候会看到目录下多了 `__pycache__ `的目录，里面存储的是 `.pyc`文件，是python脚本运行后生成的字节码缓存，主要针对 class, module 等，下次再调用的时候会检查文件是否发生了改动。



#### `if __name__ == '__main__':`
作为一个脚本，python 文件本身并不需要入口函数，解析到哪里接执行到哪里，比如：
```python
# test.py
def p(name):
    print(name)

p('rao')
```
```python
# main.py
from test import p

p('ya')
```
```pash
> python main.py
rao
ya
```
也就说，一个脚本既可以单独完成任务，也可以被其他脚本导入，在单独完成任务的时候我需要执行`p('rao')`，在被其他脚本导入的时候，我不需要执行`p('rao')`，为了满足这种特性就可以定义`if __name__ == '__main__':`，而`__name__`是一个系统变量，当你执行的是当前脚本的时候，它的值为 `__main__`，否则它的值是模块名（即文件名不带后缀）。所以优化如下
```python
# test.py
def p(name):
    print(name)

if __name__ == '__main__':
    p('rao')
```
```python
# main.py
from test import p

if __name__ == '__main__':
    p('ya')
```
```bash
> python main.py
ya
```

#### import & 模块 & 包 & `__init__.py` & `__all__`
python 中一个 .py 文件就是一个模块

```python
import test.py

from test.py import function1

from a.b import c

from a.b import (c, d, e)

from a.b.c import d
```

软件包：具有一定目录结构的代码项目，可以被导入引用。

比如有目录结构如下：`/a/b/c/d.py`，我们就需要创建`/a/b/c/__init__.py`文件，总之，如果哪个目录下的文件想要被导出，就需要创建这个文件，文件内容可以为空。

当然`__init__.py`可以写内容，重新定义导入路径，比如：`/pyflink/table/changelog_mode.py` 文件中有一个类 `ChangelogMode`，正常的导入写法是`from pyflink.table.changelog_mode import ChangelogMode`，现在可以在`__init__.py`中重新定义
```python
from __future__ import absolute_import

from pyflink.table.changelog_mode import ChangelogMode

__all__ = [
    'ChangelogMode'
]
```
然后，导入的写法变成了`from pyflink.table import ChangelogMode`。

`__all__`的作用是指明了哪些对象可以被导出。否则就是全部暴漏。

#### 字符串

```python
name = """
xxxx
"intent":"SearchBook":%s
"""

```

`r `字符串：原始字符串，如果一个字符串包含很多需要转义的字符，对每一个字符都进行转义会很麻烦。为了避免这种情况，我们可以在字符串前面加个前缀 r ，表示这是一个 raw 字符串，里面的字符就不需要转义了

```python
print(r'\(~_~)/ \(~_~)/')
```

但是`r'...'`表示法不能表示多行字符串，也不能表示包含'和 "的字符串

还可以在多行字符串前面添加 r ，把这个多行字符串也变成一个raw字符串

```python
print r'''Python is created by "Guido".
It is free and easy to learn.
Let's start learn Python in imooc!'''
```

`f `字符串：直接嵌套变量

```bash
value = (16, 'BBQ')
sql = f"insert into user values {value}"
```





#### 函数定义



#### class定义



#### 执行目录

报错 `ModuleNotFoundError: No module named 'xxxx'`

在同一个项目下是存在 `xxx`目录的，但是依然报这个错误，那是因为你执行的文件不对，虽然 python 是脚本程序，任何文件都可以执行，但是一旦构建了模块之后，就要受到一些限制，你执行的这个文件，只能 import 同级目录下的模块或者子级目录下的模块，而不能导入父级目录的模块。所以一般在根目录下创建文件去调试其他文件的代码。 
