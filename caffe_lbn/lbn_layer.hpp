#ifndef CAFFE_LBN_LAYERS_HPP_
#define CAFFE_LBN_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"

#ifdef USE_CUDNN
#include "caffe/util/cudnn.hpp"
#endif

namespace caffe {

/**
 * @brief Legacy Batch Normalization Layer, compatible with cuda_CNN
 *        Using 4 blobs for scale, shift, mean and variance, respectively
 *
 * by yangshicai on 2016.01.13
 */
template <typename Dtype>
class LBNLayer : public Layer<Dtype> {
public:
	explicit LBNLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

virtual inline const char* type() const { return "LBN"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	// spatial mean & variance
	Blob<Dtype> spatial_statistic_;
	// batch mean & variance
	Blob<Dtype> batch_statistic_;
	// buffer blob
	Blob<Dtype> buffer_blob_;
	// x_norm and x_std
	Blob<Dtype> x_norm_, x_std_;
	// Due to buffer_blob_ and x_norm, this implementation is memory-consuming
	// May use for-loop instead

	// x_sum_multiplier is used to carry out sum using BLAS
	Blob<Dtype> spatial_sum_multiplier_, batch_sum_multiplier_;

	// dimension
	int num_;
	int channels_;
	int height_;
	int width_;
	// eps
	Dtype eps_;
	// momentum factor
	Dtype momentum_;
	// whether or not using moving average for inference
	bool moving_average_;

};

}  // namespace caffe

#endif  // CAFFE_LBN_LAYERS_HPP_
