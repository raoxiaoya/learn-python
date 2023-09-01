### 环境
python-3.8.8

而 anaconda 的 archive 地址为 https://repo.anaconda.com/archive/ 但是这个地址看不出 anaconda 里对应是哪个 python 版本，  

对应的国内的镜像站为清华大学 https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=D ，根据教程，我们选择 https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.05-Windows-x86_64.exe  

- 以管理员身份运行下载的可执行文件
- 安装到 D:\ProgramData
- 添加环境变量
  - D:\ProgramData\anaconda3
  - D:\ProgramData\anaconda3\Library\mingw-w64\bin
  - D:\ProgramData\anaconda3\Library\usr\bin
  - d:\ProgramData\Anaconda3\Library\bin
  - D:\ProgramData\anaconda3\Scripts


如果是重装的 anaconda ，切记一定要重启电脑！！！

关闭翻墙软件，网络代理软件！！！


打开一个 anaconda prompt 命令行   


- 版本信息
```bash
python -V
    Python 3.8.8

anaconda -V
    anaconda Command line client (version 1.7.2)

conda info
    
     active environment : base
    active env location : d:\ProgramData\Anaconda3
            shell level : 1
       user config file : C:\Users\Administrator.DESKTOP-TPJL4TC\.condarc
 populated config files : 
          conda version : 4.10.1
    conda-build version : 3.21.4
         python version : 3.8.8.final.0
       virtual packages : __win=0=0
                          __archspec=1=x86_64
       base environment : d:\ProgramData\Anaconda3  (writable)
      conda av data dir : d:\ProgramData\Anaconda3\etc\conda
  conda av metadata url : https://repo.anaconda.com/pkgs/main
           channel URLs : https://repo.anaconda.com/pkgs/main/win-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/win-64
                          https://repo.anaconda.com/pkgs/r/noarch
                          https://repo.anaconda.com/pkgs/msys2/win-64
                          https://repo.anaconda.com/pkgs/msys2/noarch
          package cache : d:\ProgramData\Anaconda3\pkgs
                          C:\Users\Administrator.DESKTOP-TPJL4TC\.conda\pkgs
                          C:\Users\Administrator.DESKTOP-TPJL4TC\AppData\Local\conda\conda\pkgs
       envs directories : d:\ProgramData\Anaconda3\envs
                          C:\Users\Administrator.DESKTOP-TPJL4TC\.conda\envs
                          C:\Users\Administrator.DESKTOP-TPJL4TC\AppData\Local\conda\conda\envs
               platform : win-64
             user-agent : conda/4.10.1 requests/2.25.1 CPython/3.8.8 Windows/10 Windows/10.0.17134
          administrator : False
             netrc file : None
           offline mode : False

关注几个点：
python version : 3.8.8.final.0
base environment : d:\ProgramData\Anaconda3  (writable)  如果是 read only 那么后面的很多命令都需要在管理员权限运行，否则没有写权限。
administrator : False

```

### conda镜像源信息
参考 https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

conda info 的 channel URLs 中目前只有6个，我们要设置清华大学的镜像。

populated config files 为空说明还没有创建用户的配置，在windows系统上无法直接创建 .condarc 这种点号开头的文件，使用以下命令来创建
conda config --set show_channel_urls yes

编辑 C:\Users\Administrator.DESKTOP-TPJL4TC\.condarc 文件，内容为：
```bash
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
  
```

运行 conda clean -i 清除索引缓存，保证用的是镜像站提供的索引。

### pip 镜像源设置 
```bash
临时使用
	pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
	
设为默认
	升级 pip 到最新的版本 (>=10.0.0) 后进行配置：
		python -m pip install --upgrade pip
		如果您到 pip 默认源的网络连接较差，临时使用本镜像站来升级 pip：
			python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
		设为默认 pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
		Writing to C:\Users\Administrator.DESKTOP-TPJL4TC\AppData\Roaming\pip\pip.ini

```

### anaconda自带的工具
anaconda navigator 可视化界面  
anaconda prompt  命令行  
anaconda powershell prompt  命令行  
jupyter notebook  网页版编辑器，运行器，交互式  
spyder 编辑器

### 启动jupyter notebook的方式
anaconda3操作界面  
命令行输入 jupyter notebook  
系统开始菜单  

### jupyter notebook 工作目录设置
默认是在C盘的用户目录下，我们需要修改以下两个地方  
1、在命令行下执行 jupyter notebook --generate-config 得到配置文件地址，编辑此文件 c.NotebookApp.notebook_dir = 'D:/dev/php/magook/trunk/server/learn-python'  注意要将反斜线换成斜线  

2、在系统开始菜单找到jupyter notebook，右键，更多，打开文件位置，右键 Jupyter Notebook，属性，删除【目标】这一栏中双引号以及双引号中的内容，确定

### jupyter notebook 一些使用
1、在cell中输入代码，ctrl+enter 运行，或点击运行按钮，都只会运行光标所在的cell。  
2、可以设置cell的类型，代码，markdown，标题。  
3、命令模式和编辑模式。  
4、安装 nb_conda  
	conda install nb_conda  重新启动 jupyter notebook 发现发现菜单栏多了一个conda选项，
	nb_conda能够将conda创建的环境与jupyter notebook 关联，便于在使用jupyter notebook时在不同的环境下工作。 


5、更好的支持markdown文件  
conda install -c conda-forge jupyter_contrib_nbextensions   
conda install -c conda-forge jupyter_nbextensions_configurator  
重启 jpyter notebook 菜单栏增加了 nbextensions 栏目，勾选里面的 Table of contents 扩展，于是记事本界面上多了一个Table of contents 按钮， 

6、在jupyter notebook中运行终端命令  
1、在cell中以感叹号开头的命令视为终端命令。如 !dir  
2、在jupyter notebook中打开终端：File -> New -> terminal   失败？？  
关闭终端会话，Running->关闭  

7、在jupyter notebook中include 别的 .py 文件  
在cell中输入 %load 文件位置  可以引入此文件内容  
在cell中输入 %run 文件位置  可以直接运行此文件  


### conda命令与conda环境
```bash
> conda --help

usage: conda-script.py [-h] [-V] command ...

conda is a tool for managing and deploying applications, environments and packages.

Options:

positional arguments:
  command
    clean        Remove unused packages and caches.
    compare      Compare packages between conda environments.
    config       Modify configuration values in .condarc. This is modeled after the git config command. Writes to the
                 user .condarc file (C:\Users\Administrator.DESKTOP-TPJL4TC\.condarc) by default.
    create       Create a new conda environment from a list of specified packages.
    help         Displays a list of available conda commands and their help strings.
    info         Display information about current conda install.
    init         Initialize conda for shell interaction. [Experimental]
    install      Installs a list of packages into a specified conda environment.
    list         List linked packages in a conda environment.
    package      Low-level conda package utility. (EXPERIMENTAL)
    remove       Remove a list of packages from a specified conda environment.
    uninstall    Alias for conda remove.
    run          Run an executable in a conda environment. [Experimental]
    search       Search for packages and display associated information. The input is a MatchSpec, a query language
                 for conda packages. See examples below.
    update       Updates conda packages to the latest compatible version.
    upgrade      Alias for conda update.

optional arguments:
  -h, --help     Show this help message and exit.
  -V, --version  Show the conda version number and exit.

conda commands available from other packages:
  build
  content-trust
  convert
  debug
  develop
  env
  index
  inspect
  metapackage
  render
  repo
  server
  skeleton
  token
  verify
```

关于 conda environment：由于python在包管理方面的混乱，相互的依赖也很混乱，而且兼容性差，因此针对不同的项目可能需要创建不同的本地环境，这就是conda环境，conda的命令大多是需要进入一个环境的，我们使用anaconda prompt命令行会自动选择已激活的环境，而你使用别的cmd则不会，有时候会有问题（这也是 anaconda prompt 存在的意义），此时你执行 activate base 命令，那么也会进入 base 环境。  

```bash
conda env --help

usage: conda-env-script.py [-h] {create,export,list,remove,update,config} ...

positional arguments:
  {create,export,list,remove,update,config}
    create              Create an environment based on an environment file
    export              Export a given environment
    list                List the Conda environments
    remove              Remove an environment
    update              Update the current environment based on environment file
    config              Configure a conda environment

optional arguments:
  -h, --help            Show this help message and exit.
```

那么问题来了，是否在编辑器中也要来切换环境呢，不然对应的包就不存在，答案是肯定的，这其实很离谱，但是没办法，同样的道理，命令行，jupyter也要做切换再来运行。  

新创建的环境保存在anaconda3的安装目录下，比如 D:\ProgramData\anaconda3\envs  

每一个环境都有自己的一系列包，相互不影响，比如在命令行下我选择 base 环境，在编辑器我选择 tensorflow 环境，在jupyter选择 base 环境。  

vacode中切换环境：  
    方法1：编辑器右下角会展示出当前的环境，点击可选择。  
    方法2：ctrl+shift+p -> 输入 python select interpreter  
以上设置只是满足编辑器的语法解析，但是在vscode的命令行执行代码的时候还是要执行 activate tensorflow2.4 进入环境

### 包管理  
先进入一个环境再说

conda install/remove/update/search xxxx  
或者 pip install/uninstall xxx  

pip是用来安装python包的，安装的是python wheel或者源代码的包。从源码安装的时候需要有编译器的支持，pip也不会去支持python语言之外的依赖项。

conda是用来安装conda package，虽然大部分conda包是python的，但它支持了不少非python语言写的依赖项，比如mkl cuda这种c c++写的包。然后，conda安装的都是编译好的二进制包，不需要你自己编译。所以，pip有时候系统环境没有某个编译器可能会失败，conda不会。这导致了conda装东西的体积一般比较大，尤其是mkl这种，动不动几百兆甚至一G多。

然后，conda功能其实比pip更多。pip几乎就是个安装包的软件，conda是个环境管理的工具。conda自己可以用来创建环境，pip不能，需要依赖virtualenv之类的。意味着你能用conda安装python解释器，pip不行。这一点我觉得是conda很有优势的地方，用conda env可以很轻松地管理很多个版本的python，pip不行。

然后是一些可能不太容易察觉的地方。conda和pip对于环境依赖的处理不同，总体来讲，conda比pip更加严格，conda会检查当前环境下所有包之间的依赖关系，pip可能对之前安装的包就不管了。这样做的话，conda基本上安上了就能保证工作，pip有时候可能装上了也不work。不过我个人感觉这个影响不大，毕竟主流包的支持都挺不错的，很少遇到broken的情况。这个区别也导致了安装的时候conda算依赖项的时间比pip多很多，而且重新安装的包也会更多（会选择更新旧包的版本）。

最后，pip的包跟conda不完全重叠，有些包只能通过其中一个装。


### 安装 Tensorflow2.4  
tensorflow有两个版本：CPU，GPU；分别对应的是 tensorflow 和 tensorflow-gpu，后者需要安装CUDA 和 CUDNN。  

各个版本和依赖信息参考地址：https://tensorflow.google.cn/install/source_windows#cpu

```bash
1、创建环境并激活
conda create -n tensorflow2.4 python==3.8
	会将 base 环境中的包在 tensorflow2.4 环境再安装一份
	
conda activate tensorflow2.4

2、装依赖
pip install numpy matplotlib Pillow scikit-learn pandas

3、安装tensorflow

搜索一下 https://pypi.tuna.tsinghua.edu.cn/simple/tensorflow/

pip install tensorflow==2.4 

4、测试
打开python 交互命令 
>>> import tensorflow

2023-08-24 16:03:32.985731: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2023-08-24 16:03:32.991074: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
```

### 安装 Tensorflow-gpu2.4 

各个版本和依赖信息参考地址：https://tensorflow.google.cn/install/source_windows#gpu

```bash

conda create -n tensorflow-gpu2.4 python==3.8

conda activate tensorflow-gpu2.4

pip install numpy matplotlib Pillow scikit-learn pandas

pip install tensorflow-gpu==2.4 

```


### 电子书
Python - 100天从新手到大师  
https://www.cntofu.com/book/160/Day01-15/Day01/%E5%88%9D%E8%AF%86Python.md  
《Python编程从入门到实践》  
 https://pan.baidu.com/s/1o9wJq0y  密码：12od  

### 基本用法
https://github.com/phprao/learn-python

使用 jupyter notebook 来演示学习非常方便