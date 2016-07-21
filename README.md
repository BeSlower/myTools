# myTools
I keep this respository for some useful tools.
## Description
#### caffe_data_augment
- data augmentation layer source code
- color_cast      [30]
- aspect_ratio    [1.4]
- rotation        [10]
- scale_jittering [1.5]

#### caffe_normalization
- L2 and L2 normalization layer source code
- mainly add before the triplet loss layer

#### caffe_siamese
- siamese training strategy
- siamese data layer and contrasitive loss layer

#### genList
- generate proper training list for caffe image_data_layer
- for triplet loss based training  
  - one closest negitive sample (hard negtive) and one farthest positive sample to anchor
  - check out [paper](https://arxiv.org/pdf/1503.03832.pdf) for details about triplet loss
- for lifted-structured based training 
  - n closest hard negtive samples and (n-1) farthest positive samples
  - bath size is 2n (include anchor)
  - check out [paper](https://arxiv.org/pdf/1511.06452.pdf) for details about lifted-structured feature embedding

#### circular_train.sh
- circularly train caffe model shell example
- mainly for training the triplet loss based network
- include hard negtive and positive samples mining

#### convert_lmdb_to_numpy.py
- convert the format of image feature extracted from caffemodel
- convert to .npy file for further processing 
