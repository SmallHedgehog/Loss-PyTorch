## Triplet Loss

Train [LeNet](http://www.dengfanxin.cn/wp-content/uploads/2016/03/1998Lecun.pdf) using [triplet loss](https://arxiv.org/pdf/1412.6622.pdf) with PyTorch(v0.4.0)

### Methods of mine the triplets

See the website "https://zhuanlan.zhihu.com/p/35560666" for more details.

* Offline triplet mining

Pass

* Online triplet mining

Select hardest positive and hardest negative of each anchors.

### Result

* Select triplet examples randomly

Same setup, training twice.
![randomly](https://github.com/SmallHedgehog/Loss-PyTorch/blob/master/triplet_loss/pics/randomly.png)

* Online triplet mining of batch hard[1][2]

Same setup, training twice.
![batch hard](https://github.com/SmallHedgehog/Loss-PyTorch/blob/master/triplet_loss/pics/batch%20hard.png)

### Reference
[1] mountain blue. https://zhuanlan.zhihu.com/p/35560666
[2] Alexander Hermans, Lucas Beyer and Bastian Leibe, Visual Computing Institute RWTH Aachen University. In Defense of the Triplet Loss for Person Re-Identification.
