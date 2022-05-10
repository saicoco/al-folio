---
layout: distill
title: Deep-Feature下的图像检索
description: 从大局出发看用deep-feature做图像检索
date: 2022-05-09
categories: 检索
comments: true
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
bibliography: 2022-05-09-distillImageRetrieval.bib
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

### 图像检索在做什么  
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

但是对于离线特征的构建方式，需要考虑的一个问题是,**如何使得这些特征或者这类构建方式能够适应数据分布的变化。**因为特征抽取网络是在特定的数据上训练得到的，基于该模型进行特征抽取，对于与训练数据分布差异较大的样本，特征未必可用于当前数据。因此涌现出在任务数据上进行finetune的方法，使得当前特征提取模块可以适应当前数据，同时配合最终的metric learning，提升检索性能。  

而近两年自监督学习让大家认识到，自监督任务是属于instance-level的学习方式，因此有工作将自监督学习适配到检索任务上。同时配合更为强力的数据增强，结合局部特征抽取等方法，将检索性能进一步提升。  

以上是对整个大纲的概述，而对于整个检索算法组成而言，下图可以看作是比较完整的一个概括。

<div class="fake-img l-page">
  <p>{% include figure.html path="https://tva1.sinaimg.cn/large/e6c9d24ely1h23inwvazjj22b30u0n33.jpg" class="img-fluid rounded z-depth-1" %}</p>
</div>
<div class="caption">图像检索大致流程：包含特征提取、特征聚合、rank等
</div>
接下来，我们主要针对离线特征、在线finetune等不同的特征构建方式进行概括描述。

***
### 用离线特征做图像检索
用离线特征做图像检索，指的是利用离线抽取好的特征作为图片的基础特征，再次基础上构建检索算法。因此离线特征的质量对后续的检索算法的至关重要。在CNN刚起步的阶段，大家采用CNN最后一层输出作为离线特征，在使用过程中发现，这类特征语义性较强，与类别相关，但是对于instance级别的区分性较差，同时丢失空间结构信息。于是大家开始寻找能够捕捉空间结构信息的方法。

#### 关键特征的构造与提取
最近的一些region-based的特征提取方法，大致可以分为feature-level、image-level两种。前者主要在CNN的feature空间维度做关键特征的提取，后者主要通过image输入层处理，通过挖掘图片的局部区域，得到局部特征，这里我们主要描述前者，具体可以用下图展现：
<div class="fake-img l-page">
  <p>{% include figure.html path="https://tva1.sinaimg.cn/large/e6c9d24ely1h23lwtook3j20sv1dpdju.jpg" class="img-fluid rounded z-depth-1" %}</p>
</div>
<div class="caption">关注局部特征的抽取方法</div>
上述特征总结如下：
- MAC与R-MAC：目的在于抽取每个通道内的最大响应值作为当前区域的关键点特征。R-MAC的不同之处在于，将图片切分为多个块，对每一块区域进行最大响应值提取，最终将所有的区域、通道得到的特征拼接，得到最终的离线特征。R-MAC结合CNN，在不同感受野下抽取图片的关键特征，使得特征有很强的判别性。但是R-MAC最终抽取得到的特征与区域块数*通道数有关，当块数以及通道很多的时，特征维度会对应的增加，因此需要PCA等相关的降维操作减少特征的冗余。另外R-MAC抽取特征较为局部，所以对于图片全局的表达能力较局部弱，因此需要搭配对应的特征聚合模块提升特征的表达能力。
- GeM：是一种比较general的pooling方法，对当前通道内的特征值进行指数变换，将响应较大的特征变得更明显，将响应较低的特征抑制，进而达到提取关键特征的目的。GeM可以提取全局下的关键特征，较R-MAC有一定的优势；GeM的指数因子p需要人为的调整，因此需要不断的观察做出调整，因此后续的工作在系数上做了一些改动。
- SPoC、CroW、CAM+CroW：思路都是为当前的像素值乘以一个比较合适的系数，让应该响应大的特征更为突出。如以距离图像中心的高斯距离为系数、为不同的空间位置获取不同的系数、通过特征响应图得到attention map,作为每个位置的响应。

上述的特征可以看作是对CNN已有的特征做进一步加工改造，使得特征可以用作检索任务。但是仅仅利用这些特征去做检索是不够的，这些特征可以看作是基础的特征，直接用作检索存在存储、效率的问题。

为了解决特征的高维度、高冗余带来的存储、检索效率问题，一些工作尝试在聚合、降维等方面对特征进行处理。离线方法如PCA提前将特征降维，在线方法如VLAD系列方法：netVLAD等通过在线聚类的方式，将高维特征映射为这些聚类中心的相关表示。此处涉及公式较多，后续涉及到详细进行分析。

***
### 端到端fine-tune做图像检索
基于离线特征的检索方法比较直接，寻找开源的模型作为特征提取器，将离线特征存储好作为检索网络的训练、前向等。但是离线特征存在一些问题：
- 作为特征提取器，是通过公开数据的训练得到。进而使得最终抽取得到的特征与训练数据具有相关性。如在狗相关的数据训练得到的分类模型，在图片上的高响应区域将会是与狗相关或者类似的区域，如黑色、像鼻子，像狗子的区域。如果训练数据与检索数据存在较大的差异，此时得到的特征可用性不大。【ImageNet有1000类，基本可以覆盖生活中的大类，因此大部分开源模型做离线特征抽取得到的特征不会太拉垮】但是总有例外，如细粒度等场景
- 即使公开数据与检索数据差异不大，但是离线特征提取相当于将这部分网络冻结，使得这部分得到的特征与应用场景终究有一定的差异。

因此有一些工作尝试通过finetune的方式，提升检索性能。finetune的方式大致可以分为两类，一种为分类为主，即通过在目标数据上进行分类任务的finetune，使得当前模型可以关注到当前数据上的关键信息；另外一种类似metric learning，通过构造正负样本对，学习正样本应该具备的关键特征，进而间接提升检索性能。显而易见，以分类为主的方法，只能学习到区分类别的特征，对于类内样本的区分性不够，因此学习得到的特征较metric learning的方式较弱。大致的做法可以用下图说明：
<div class="fake-img l-page">
  <p>{% include figure.html path="https://tva1.sinaimg.cn/large/e6c9d24ely1h23kvaf2dqj213u0u0q7c.jpg" class="img-fluid rounded z-depth-1" %}</p>
</div>

<div class="caption">classification-based与vertification-based方法汇总
</div>
上图除图a，图g为分类任务，后续的任务均在分类任务上加入了metric learning的任务。可以看作是在分类任务的监督下，模型可以关注到类别间的区分特征，在此基础上，加入instance-level的metric learning，使得模型能够关注到更为区分性的特征。这里可实例分割更像，coarse-to-fine。  

***
以上是从Deep Image Retrieval:A Survey<d-cite key="2021Deep"></d-cite>中截取总结的一些内容，更为详细的内容可以阅读文章。文章主要总结了自监督学习之前的图像检索相关的工作以及流程，而自监督学习之后的图像检索任务则与自监督任务十分相关。因为自监督学习为instance-level的学习方式，所以构造正负样本配合自监督学习，可以迁移至检索任务。另外，以上虽然分为离线特征与在线特征两种方式去描述deep-feature如何做图像检索，但是离线特征的方式也可以用在在线finetune阶段，只不过是在不同阶段提出的主要做法。后续会记录一些自监督学习在视频检索、图像检索的工作，算是阶段性的记录，后续翻出来还能回忆起其中的关键点。就写到这里吧～


