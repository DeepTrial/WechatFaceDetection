# 基于微信的物体识别系统

此项目是2017年第七届“华为杯”中国大学生智能科学竞赛全国二等奖、华为专项奖作品。

## 介绍

通过本系统 你可以在微信上远程训练一个单一目标的物体识别系统，（默认是人脸检测），你也可以删除训练好的模型，自己上传训练集训练，由于使用了数据增强，你可以不需要传入太多的图片就可以实现识别的功能。对于识别错误的目标，你可以发送语音消息：错误，或者发送文本消息错误，来启动在线学习，纠正模型。


![](https://i.imgur.com/JgTDo0L.png)

## 使用界面截图
<img src="https://i.imgur.com/nQ7WAEe.png" width="500" hegiht="313" align=center />
<img src="https://i.imgur.com/d9P3LM8.png" width="500" hegiht="313" align=center />


## 配置要求

python3.5.2（版本必须一致，建议安装anaconda）

tensorflow

itchat（微信端框架）

opencv3

pywave

ffmpeg

使用过程需要联网

## 使用指南

运行starter，扫码登录微信，扫码的微信号就是服务器啦，你可以向它发送消息，试试！

输入介绍，可以看到系统的简介和所有关键词。

## 原理简介

我们的产品基于生活中常用的微信平台进行交互，操作方便、简单，用户几乎不需要学习就可以熟练操作。产品的核心是基于深度学习的图像识别模型，我们采用了实际效果非常优异的VGG19模型[6]、用户可以快速、准确获取识别结果。对于识别错误的图像，用户只需要向系统回复“错误”，模型会自动启动，重新学习特征。随着使用时间的增长，本产品准确率会越来越高。当用户的需求发生改变时，只需要向产品回复“删除训练集”，并重新上传想要识别的目标图片，产品会开始自主学习，最终识别出新的目标。由于使用了数据增强技术，产品会依据当前样本实时变形出新样本、提升数据量，所以用户并不需要上传太多的目标图片，就可以实现较高的准确率。

### 系统方案设计

<img src="https://i.imgur.com/FQxQNLL.png" width="500" hegiht="313" align=center />

系统依据功能整体划分为2个部分：

- 微信交互处理：信息接收、关键词过滤、自然语言处理、人机交互；
- 图形处理识别：图形存储、OpenCV检测、深度学习检测、在线学习、自动回复。

这里选择的深度学习模型是VGG19模型，OpenCV的作用在人脸检测时处理简单样本，让难以判断样本交给深度学习模型处理，加快检测速度。

### 软件流程图

![](https://i.imgur.com/xdM8BjZ.png)
左侧为系统的总流程，在用户扫码登录系统之后即开始业务流程，用户终止程序运行后退出即时通信流程。在即时通信流程中对三种消息进行处理，并统一以文字信息的方式回复。在即时通信流程中，红色边框的模块代表特殊的关键词会影响这些模块。

## 训练数据下载
默认的人脸检测的数据集 包含通用负样本
链接：https://pan.baidu.com/s/1i5aFF6l 密码：dasp


