---
layout: distill
title: 2024-10月随想
description: 写到哪就到哪
date: 2024-10-22
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

大模型输出控制输出json格式，可以使用StrictJson，这个是从TaskGen这篇文章中看到的，github地址是：https://github.com/simbianai/taskgen

## TaskGen

### 做一个agent需要注意的地方
要想做一个通用的agent，agent就需要适配各种各样的任务，可以通过配置通用常见的函数或者方法，来组合满足不同的任务诉求。而设计这样的agent，需要实现几点：
1. 任务理解后的内容输出格式保持一致，满足不用任务接口间的对齐，同时可以减少token的浪费
2. 避免递归，将每个子任务交给一个单独的函数或方法处理
3. 信息共享，不同子任务之间信息传递，避免生成的结果有偏

agent的任务是能够为当前任务挑选合适的子函数或者方法进行处理，两种处理方式：
1. 采用CoT的方式理解当前任务应该使用的函数
2. 根据函数的入参选择需要使用的函数

### 如何精准的输出json格式
"Given a similar input prompt, asking the LLM to output in a JSON format generally gives much less verbose output as compared to free text", 告诉模型的输出json格式，可以实现输出json内容。但是有时候会失效。
文章采用StrictJSON实现稳定输出json格式（感觉我可以尝试一下， https://github.com/tanchongmin/strictjson）


### 重要的组件
- agent需要模块化的组件以及鲁棒的函数，处理子任务数据；
- 另外需要共享不同任务间的状态，有两种方式实现，其一为保存子任务的输出，作为后续任务的prompt中的内容；另外一种为通过python共享变量实现，在不同函数中共享，无需输入至prompt中
- context内容的保存，主要用来获取任务相关的信息，如寻找文本中的xxx内容

### 所以TaskGen可以干什么
1. 自定义实现agent，支持函数定义
2. 约束json输出
3. 支持多个agent调用