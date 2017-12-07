#PspNet 

PspNet is in Caffe

This repo attempts to reproduce this amazing work by  : 
[PspNet](https://arxiv.org/pdf/1612.01105.pdf)

## Requirements

- [Cuda-8.0 (>= 1.0.0)](https://www.tensorflow.org/install/install_linux)
- [Opencv-3.1.0]



##How T0
1. Go to `./caffe-pspnet` folder, make `build`folder and run cmake ..  and then make - j[Number of cores].
2. Go to `./pspnet_test` folder, open CMakeLists.txt and mention caffe build path.
3. Go to `./pspnet_test` folder make build folder and run cmake .. and then make all.



## For Training
1. Go to `./utils/prototxt_training/solver_PSP.prototxt` Mention the network prototxt and snapshot path.
2. 

