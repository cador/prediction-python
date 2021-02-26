# prediction-python

本人新书《Python预测之美 : 数据分析与算法实战》已于2020年7月1日印刷出版，欢迎关注。

京东链接：<a href="https://item.jd.com/12685261.html" target="_blank">点击打开</a>。

<img src="http://image.cador.cn/resource/05efe48a5011f926.jpg" />

《Python预测之美-数据分析与算法实战》简介

Python 是一种面向对象的脚本语言，其代码简洁优美，类库丰富，开发效率也很高，得到越来越多开发者的喜爱，广泛应用于Web 开发、网络编程、爬虫开发、自动化运维、云计算、人工智能、科学计算等领域。预测技术在当今智能分析及其应用领域中发挥着重要作用，也是大数据时代的核心价值所在。随着AI 技术的进一步深化，预测技术将更好地支撑复杂场景下的预测需求，其商业价值不言而喻。基于Python 来做预测，不仅能够在业务上快速落地，还让代码维护更加方便。对预测原理的深度剖析和算法的细致解读，是本书的一大亮点。

本书共分为3 篇。第1 篇介绍预测基础，主要包括预测概念理解、预测方法论、分析方法、特征技术、模型优化及评价，读者通过这部分内容的学习，可以掌握预测的基本步骤和方法思路。第2 篇介绍预测算法，该部分包含多元回归分析、复杂回归分析、时间序列及进阶算法，内容比较有难度，需要细心品味。第3 篇介绍预测案例，包括短期日负荷曲线预测和股票价格预测两个实例，读者可以了解到实施预测时需要关注的技术细节。希望读者在看完本书后，能够将本书的精要融会贯通，进一步在工作和学习实践中提炼价值。

### 如何搭建环境

第一步，安装Anaconda

您可参考 Anaconda 官网说明来安装 Anaconda，地址为：`https://www.anaconda.com/products/individual#download-section`

直接下载的链接地址：

 - Windows 64-Bit Graphical Installer (457 MB), `https://repo.anaconda.com/archive/Anaconda3-2020.11-Windows-x86_64.exe`
 - Windows 32-Bit Graphical Installer (403 MB), `https://repo.anaconda.com/archive/Anaconda3-2020.11-Windows-x86.exe` 
 - MacOS 64-Bit Graphical Installer (435 MB), `https://repo.anaconda.com/archive/Anaconda3-2020.11-MacOSX-x86_64.pkg`
 - MacOS 64-Bit Command Line Installer (428 MB), `https://repo.anaconda.com/archive/Anaconda3-2020.11-MacOSX-x86_64.sh`
 - Linux 64-Bit (x86) Installer (529 MB), `https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh`
 - Linux 64-Bit (Power8 and Power9) Installer (279 MB), `https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-ppc64le.sh`

第二步，自定义Python环境

`conda create -n prediction python=3.8.5`

第三步，安装依赖库

`pip install -r requirements.txt`

在国内，如果下载慢的话，可以用以下命令

`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
