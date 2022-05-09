---
layout: distill
title: Deep-features下的图像检索做什么
description: 回顾以前的基于deep-feature图像检索工作，为后续视频检索相关工作做铺垫
date: 2022-05-09
categories: 检索
authors:
  - name: saicoco

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: 图像检索在做什么
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: 用离线特征做图像检索
  - name: 端到端fine-tune做图像检索
  - name: 自监督的方式做图像检索


# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

---

**写在前面的话:**
最近在做视频检索相关的任务，在查找相关文献时，自然的转向图像检索，进而发现图像检索领域涉及内容很多，于是找到近期的survey进行阅读。文章主要讲述与deep-feature相关的方法以及存在的问题，于是进行摘要总结。文章中划分方式较细，此处没有参考文章中的划分方式，而是采用较为粗略的划分，从特征的获取方式入手进行划分。同时，对于涉及到的具体方法对应的文章不做讲述分析，后续涉及到具体方法的使用时再进行详细的描写

## 图像检索在做什么  
图像检索，是指拿一张图片，去某个图片库里找到你想要的图片。**那么问题来了，什么叫想要的图片？** 以这张二哈为例子
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://tva1.sinaimg.cn/large/e6c9d24ely1h22fjua9n1j20t70mrmy0.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://tva1.sinaimg.cn/large/e6c9d24ely1h22fjua9n1j20t70mrmy0.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://tva1.sinaimg.cn/large/e6c9d24ely1h22iiaony2j20p80ylt9l.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">图1与图2为instance-level相似；图1与图3为category-level相似
</div>
拿着这张图片去图片库中进行检索时，无外乎以下两种：
- 查找和这样图片一模一样的图片，只不过图片大小、分辨率不同，或者该图为图库中某张图片的一部分
- 查找和这张图片同类型的图片，背景、颜色可以不同，但是必须类别为狗或者哈士奇等，类别相同

以上两种分类，第一种叫做instance-level, 必须实例相同，近似或者相同；第二种叫做category level，“类别”相同即可。相比类别相似的检索，instance-level的检索更具有挑战性，核心的问题主要聚焦于**如何构建具有较强的具有判别性的特征**以及**如何构建具有可迁移性的特征**，这其中主流分为离线特征、在线特征；而从优化方法上，又可以分为监督与“无”监督两种。为了更好的描述这部分内容，采用特征构建的方式进行分别介绍。整体的大纲可以如下图所示：
<div class="fake-img l-body">
  <p>{% include figure.html path="https://tva1.sinaimg.cn/large/e6c9d24ely1h22hixx9obj21780h0t9l.jpg" class="img-fluid rounded z-depth-1" %}</p>
</div>
在VGG等深度网络发布之后，基于深度网络的特征用于图像检索，做法往往直接抽取最后一层fc输出的特征作为检索的特征，具有全局性，但是局部性较差。另外由于网络为分类网络，此时的特征语义性较强，但是对于局部信息的表达较弱。
针对这个问题，后续的工作提出了空间特征挖掘的方法，如R-MAC, GeM, SPoC等类似的方法，同时也有在输入层面构造局部信息输入的方法构建局部特征，如RPN，select-search等。这类方法开始在离线特征抽取的方法中应用较多，多基于CNN构建此类离线抽取的特征，用于后续的检索网络。
然而过分关注局部特征忽略全局表达也是存在一定问题的，进而涌现出一批聚合局部特征或者融合局部-全局特征的工作，如netVLAD等聚合方法，将局部特征聚合为更为紧凑的特征表达，提升特征的表达能力。
但是对于离线特征的构建方式，需要考虑的一个问题是,**如何使得这些特征或者这类构建方式能够适应数据分布的变化**因为特征抽取网络是在特定的数据上训练得到的，基于该模型进行特征抽取，对于与训练数据分布差异较大的样本，特征未必可用于当前数据。因此涌现出在任务数据上进行finetune的方法，使得当前特征提取模块可以适应当前数据，同时配合最终的metric learning，提升检索性能。
而近两年自监督学习让大家认识到，自监督任务是属于instance-level的学习方式，因此有工作将自监督学习适配到检索任务上。同时配合更为强力的数据增强，结合局部特征抽取等方法，将检索性能进一步提升。

以上是对整个大纲的概述，让我们对整个发展过程有一个全局的认识。接下来，我们就分块对以上的内容进行描述。

***