﻿### 1.说明
项目主要模拟了对于联邦学习中，用非独立同分布的数据来训练Diffusion
需要说明的是，repo以展示思想为主，并没有用到很好的数据集。浅尝一下labml.ai封装的Diffusion库
### 2.环境搭建
    pip install labml-nn
    pip install avalanche-lib
### 3.run
用联邦的思想训，其中有模拟客户和中心服务器
`python federated.py`
单纯的用一个服务器/用户训
    `python centralized.py`

### 4.数据集和Epoch设置

在function main().中修改默认配置

Eg：

    experiment.configs(configs, {
        'dataset': 'MNIST',  # 'MNIST','CelebA'
        'image_channels': 1,  # 1,3
        'epochs': 5,  # 5,100
    })

### 5.采样
对于一个训好的模型，这里可以做Sample
    `python evaluate.py`

### 6.文件解释
`federated.py`和`fedavg.py`分别表示客户训练和中心分发的行为<br>
`improved_fed`,`imporved_avg`是针对原方法加了一点点优化，其实就是联邦学习中根据不同用户训练效果设置的惩罚
### 7.补充
[2024-12] 这个框架在很多层面的封装程度不如Diffuiser，扩散模型还是得大公司来整
