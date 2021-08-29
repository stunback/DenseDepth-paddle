# DenseDepth-paddle
[中文说明](README_cn.MD)    
This is a project to reproduce DenseDepth using paddlepaddle.    

飞桨论文复现挑战赛（第四期）  
论文题序号5   

High Quality Monocular Depth Estimation via Transfer Learning    的 Paddle2.1 实现   

# 使用说明   
## 环境要求  
AIStudio 默认的环境就行  
paddlepaddle-gpu>=2.0  
pillow  
opencv  

## 训练好的模型  
由于模型比较大，因此放在百度网盘中，链接如下：  
链接：https://pan.baidu.com/s/1f1lYptz3xVMs3mJKvVgrUw   
提取码：as0z   
## 预训练模型
densenet预模型链接如下：     
链接：https://pan.baidu.com/s/1KUPnjUgpG40VSDLBHEcRIQ    
提取码：zid4   

## 数据集   
将DenseDepth作者给出的数据集放在本repo的dataset/下  
原数据集链接：https://drive.google.com/drive/folders/1TzwfNA5JRFTPO-kHMU___kILmOEodoBo?usp=sharing

## 代码  
train.py 训练代码  

test.py  测试代码  

evaluate.py  精度评估代码 

## 训练方法
由densenet迁移学习，将densenet预训练模型放在weights/中，然后开始训练：     
python train.py     
 
## 精度评估   
python evaluate.py   

本地评估结果  
       a1,         a2,         a3,        rel,        rms,     log_10  
     0.895,      0.981,      0.995,      0.106,      0.456,      0.044


## 实际深度估计效果测试
把需要测试的图片添加到examples/中   
python test.py