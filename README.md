#PspNet 

- Author: Manish Soni Email: manishsoni1908@gmail.com 
- License: license

PspNet is in Caffe

This repo attempts to reproduce this amazing work by Hengshuang Zhao  : 
[PspNet](https://arxiv.org/pdf/1612.01105.pdf)

## Requirements

- [Cuda-8.0 (>= 1.0.0)]
- [Opencv-3.1.0]



##How To
1. Go to `./caffe-pspnet` folder, make `build`folder and run cmake ..  and then make - j[Number of cores].
2. Go to `./pspnet_test` folder, open CMakeLists.txt and mention caffe build path.
3. Go to `./pspnet_test` folder make build folder and run
 - $ cmake .. 
 - $ make all.



## For Training
1. Go to `./utils/prototxt_training/solver_PSP.prototxt` Mention the network prototxt, snapshot path and max_iteration.
2. Go to `./utils/prototxt_training/original_pspnet.prototxt` Mention the NumClasses in `conv6_apc17` and `conv6_1_apc17` layer.
3. Go to `./caffe-pspnet/build/tools` Run command 
  - $ ./caffe train -gpu all -solver `[Path to solver.prototxt]` -weights`[path to pre-tarined model]`



## For Testing
1. Go to `./pspnet_test/build` Run command 
  - $ ./test `[path of testing prototxt]` `[path of trained model]` `[path of test image directory]`
2. All done



##Results

![Original Image](https://github.com/ManishSoni1908/PspNet-Caffe/blob/master/results/img_0.png)
![Segmented output](https://github.com/ManishSoni1908/PspNet-Caffe/blob/master/results/seg_0.png)

 
