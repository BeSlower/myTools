#include <algorithm>
#include <vector>

#include "caffe/layers/contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    
    template <typename T>
    static bool ComputeMargin(const T *dist,
                              const T *labels_a,
                              const T *labels_b,
                              const int n,
                                T &margin)
    {
        int num_pos = 0;
        T m_margin;

        vector<pair<T, bool> > dist_vector;
        vector<bool> similarity_labels(n,0);

            for (int i = 0; i < n; ++i)
            {
                if (labels_a[i] == labels_b[i])
                {
                    similarity_labels[i] = 1;
                }
                else
                {
                    similarity_labels[i] = 0;
                }
                dist_vector.push_back(make_pair(dist[i], similarity_labels[i]));
            }
        sort(dist_vector.begin(), dist_vector.end(), less<pair<T, bool> >());

        for (int i = 0; i < n; ++i)
        {
            num_pos += static_cast<int>(similarity_labels[i]);
        }

        if (num_pos == 0 || num_pos == n)
        {
            m_margin = dist_vector.back().first;
        }
        else
        {
            int count_pos = 0;
            int count_neg = 0;
            int min_err = num_pos;

            m_margin = T(0);

            for (int i = 0; i < n; ++i)
            {
                if (dist_vector[i].second)
                {
                    count_pos++;
                }
                else
                {
                    count_neg++;
                }

                //choose a threshhold, which makes less false positives
                int err = num_pos - count_pos + count_neg;
                if (err < min_err)
                {
                    min_err = err;
                    m_margin = dist_vector[i].first;
                }
            }
        }

        margin = m_margin;
        return true;
    }
	template <typename T>
	static bool L2_norm(T	*array,
		const int n)
	{
		T sum = T(0);
		for (int i = 0; i < n; i++)
		{
			sum += array[i] * array[i];
		}
		sum = 1.f / std::max(1e-6f, std::sqrtf(sum));

		for (int i = 0; i < n; i++)
		{
			array[i] = array[i] * sum;
		}
		return true;
	}


template <typename Dtype>
void ContrastiveLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  CHECK_EQ(bottom[3]->channels(), 1);
  CHECK_EQ(bottom[3]->height(), 1);
  CHECK_EQ(bottom[3]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  const int channels = bottom[0]->channels();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
  
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  Dtype loss(0.0);
  
  //ComputeMargin
  for (int i = 0; i < bottom[0]->num(); ++i) {
      dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
          diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
  }
  ComputeMargin(dist_sq_.cpu_data(), bottom[2]->cpu_data(), bottom[3]->cpu_data(), bottom[0]->num(), margin);
  margin = sqrt(margin);
  global_margin_ = margin;
  //
  for (int i = 0; i < bottom[0]->num(); ++i) {
//     dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
//         diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
    //add
    //int label1 = static_cast<int>(bottom[2]->cpu_data()[i]);
    //int label2 = static_cast<int>(bottom[3]->cpu_data()[i]);
    //printf("label1 = %d, label2 = %d\n", label1, label2);
    //printf("dist_sq = %.5f\n", dist_sq_.cpu_data()[i]);
    //add
    if (static_cast<int>(bottom[2]->cpu_data()[i]) == static_cast<int>(bottom[3]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      if (legacy_version) {
        loss += std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0));
      } else {
        Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq_.cpu_data()[i]),
          Dtype(0.0));
        loss += dist*dist;
      }
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.contrastive_loss_param().margin();
  margin = global_margin_;
  bool legacy_version =
      this->layer_param_.contrastive_loss_param().legacy_version();
  for (int i = 0; i < 4; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(bottom[i]->num());
      int num = bottom[i]->num();
      int channels = bottom[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = bottom[i]->mutable_cpu_diff();
        if (static_cast<int>(bottom[2]->cpu_data()[j]) == static_cast<int>(bottom[3]->cpu_data()[j])) {  // similar pairs
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
        } else {  // dissimilar pairs
          Dtype mdist(0.0);
          Dtype beta(0.0);
          if (legacy_version) {
            mdist = margin - dist_sq_.cpu_data()[j];
            beta = -alpha;
          } else {
            Dtype dist = sqrt(dist_sq_.cpu_data()[j]);
            mdist = margin - dist;
            beta = -alpha * mdist / (dist + Dtype(1e-4));
          }
          if (mdist > Dtype(0.0)) {
            caffe_cpu_axpby(
                channels,
                beta,
                diff_.cpu_data() + (j*channels),
                Dtype(0.0),
                bout + (j*channels));
          } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ContrastiveLossLayer);
#endif

INSTANTIATE_CLASS(ContrastiveLossLayer);
REGISTER_LAYER_CLASS(ContrastiveLoss);

}  // namespace caffe
