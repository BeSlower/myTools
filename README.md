# myTools
I keep this respository for some useful tools.
## Description
#### caffe_ccl
- coupled-clusters loss layer source code implemented by [nicklhy](https://github.com/nicklhy/caffe-dev)
- check out [paper](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Relative_Distance_CVPR_2016_paper.pdf) for detail about coupled-clusters loss

#### caffe_data_augment
- data augmentation layer source code
- color_cast      [30]
- aspect_ratio    [1.4]
- rotation        [10]
- scale_jittering [1.5]

#### caffe_lbn
- batch normalization source code
- check out [paper](http://arxiv.org/pdf/1502.03167.pdf) for detail about batch normalization

#### caffe_normalization
- L1 and L2 normalization layer source code
- mainly use in front of the triplet loss layer

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
- include hard negative and positive samples mining

#### convert_lmdb_to_numpy.py
- convert the format of image feature extracted from caffemodel
- convert to .npy file for further processing 
