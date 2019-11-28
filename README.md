# SV-X-Softmax
SV-X-Softmax is a new loss function, which adaptively emphasizes the mis-classified feature vectors to guide the 
discriminative feature learning. For more details, one can refer to our paper: \
"Mis-classifided Vector Guided Softmax Loss for Face Recognition" [arxiv](http://www.cbsr.ia.ac.cn/users/xiaobowang/papers/AAAI2020.pdf) \
In AAAI Conference on Artificial Intelligence (AAAI) 2020, **Oral** Presentation. \
![alt](https://github.com/xiaoboCASIA/SV-X-Softmax/blob/master/figure1.png) \
Thank **Shifeng Zhang** and **Shuo Wang** for their helpful discussion and suggestion.

## Introduction
This is an implementation of our SV-X-Softmax loss by **Pytorch** library. The repository contains the fc_layers.py and loss.py
The old version: "Support Vector Guided Softmax Loss for Face Recognition" [arxiv](https://arxiv.org/abs/1812.11317) \
is implemented by **Caffe** library and does not remove the overlaps between training set and test set. The performance comparsion 
may not be fair in the old version.

## Dataset
The original training set is [MS-Celeb-1M-v1c](http://trillionpairs.deepglint.com/overview), which constains 86,876 identities.
However, in face recognition, it is very important to perform open-set evaluation, i.e., there should be no overlapping identities 
between training set and test set. In that way, we use the publicly available [script](https://github.com/happynear/FaceDatasets) to remove 14,186 identities from the training 
set MS-Celeb-1M-v1c. For clarity, we donate the refined training dataset as 
[MS-Celeb-1M-v1c-R](https://github.com/xiaoboCASIA/SV-X-Softmax/blob/master/deepglint_unoverlap_list.txt). 

## Architecture
The AttentionNet-IRSE network used in our paper is derived from the papers:
1. "Residual attention network for image classification" [Paper](https://arxiv.org/abs/1704.06904?source=post_page---------------------------)
2. "Arcface additive angular margin loss for deep face recognition" [Paper](https://arxiv.org/abs/1801.07698)


## Others
1. Note that our new loss is based on the well-cleaned training sets, when facing new datasets, one may need to clean them. 
2. On the small test set like LFW, the improvement may not be obvious. It may be better to see the comparision on MegaFace or more large-scale test set. 
3. Both training from stratch and finetuning are ok. One may try more training strategies.
4. We won the **1st** place in [RLQ](https://www.forlq.org/) challenge (all four tracks) and **2st** place in [LFR](http://www.insightface-challenge.com/overview) challenge (deepglint-large track)

## Citation
If you find SV-X-Softmax helps your research, please cite our paper:
