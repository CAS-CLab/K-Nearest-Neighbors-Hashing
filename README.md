# K-Nearest-Neighbors-Hashing
Matlab implementation of "K-Nearest Neighbors Hashing".

### Dependencies :
* **MatLab >= R2016b** (GPU support knnsearch)
* MatLab <= R2016a (CPU only knnsearch)

### How to use ?

### Datasets :
OneDrive | gist | vggfc7 | uint | alexnet
------------ | ------------- | ------------- | ------------- | -------------
Cifar10 | ✓  | ✓  | ☐  | ☐ 
MNIST |✓ | ☐  | ✓ | ☐ 
Labelme | ✓ | ✓ | ☐  | ☐ 
Places205 | ☐  | ☐  | ☐  | ✓

Baidu Pan | gist | vggfc7 | uint | alexnet
------------ | ------------- | ------------- | ------------- | -------------
Cifar10 | ✓  | ✓  | ☐  | ☐ 
MNIST |✓ | ☐  | ✓ | ☐ 
Labelme | ✓ | ✓ | ☐  | ☐ 
Places205 | ☐  | ☐  | ☐  | ✓

`uint` refers to MNIST 784-D (28x28) gray-scale feature vector, which is represented by uint8.

### Brief Intro :
<img src="./img/KNNH.png" width="700" height="240" />

MSE may not lead to the best binary representation. We propose to use the conditional entropy as the criterion.
<img src="./img/KNNH2.png" width="700" height="230" />

### Acknowlegment :
We would like to thank the authors of [Gemb](https://github.com/hnanhtuan/Gemb) and [MiHash](https://github.com/fcakir/mihash) for sharing their codes! This project is built on previous methods such as [ITQ](http://www.cs.unc.edu/~lazebnik/publications/cvpr11_small_code.pdf), [BA](https://arxiv.org/abs/1501.00756), [KMH](http://kaiminghe.com/publications/cvpr13kmh.pdf), [SH](https://papers.nips.cc/paper/3383-spectral-hashing), [PCAH](http://www.ee.columbia.edu/ln/dvmm/publications/12/PAMI_SSHASH.pdf) and [SPH](https://sglab.kaist.ac.kr/Spherical_Hashing/Spherical_Hashing.pdf).

### References :
```bib
@article{KNNH,
  author    = {Xiangyu He and Peisong Wang and Jian Cheng},
  title     = {K-Nearest Neighbors Hashing},
  booktitle = {2019 {IEEE} Conference on Computer Vision and Pattern Recognition},
  month     = {July},
  year      = {2019}
}
```
