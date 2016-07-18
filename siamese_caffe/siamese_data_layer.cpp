#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/siamese_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

typedef struct _PAIR_SAMPLE_
{
	int smp_idx;                //
	int smp_label;
	int pair_idx;               //
	int pair_label;
	int label;                  // positive pairs: 1 ; negtive pairs: 0
}PAIR_SAMPLE;

namespace caffe {

template <typename Dtype>
SiameseDataLayer<Dtype>::~SiameseDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void SiameseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  const int interpolation = this->layer_param_.image_data_param().interpolation();
  const int resize_mode = this->layer_param_.image_data_param().resize_mode();

  CHECK((new_width >= 0) && (new_height >= 0))
            << "Current implementation requires new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
//  string filename;
//  int label;
//  while (infile >> filename >> label) {
//    lines_.push_back(std::make_pair(filename, label));
//  }

  string line_buf, filename, label_str;
  int tmp_i = 0;
  while (!(getline(infile, line_buf).fail())) {
    int    label;
    size_t pos;
    stringstream  label_stream;
    //tmp_i++;
    line_buf.erase(line_buf.find_last_not_of(" \r\n") + 1);
    pos = line_buf.find_last_not_of(" 0123456789");

    //LOG(INFO) << "Opening file " << tmp_i << ", path: " << line_buf;
    //LOG(INFO) << "pos: " << pos;

    //LOG(INFO) << "line_size: " << line_buf.size();
    filename = line_buf.substr(0, pos + 1);

    //LOG(INFO) << "filename: " << filename;

    label_str = line_buf.substr(pos + 2);
    label_stream.str(label_str);
    label_stream >> label;
    //LOG(INFO) << "label :" << label;
    lines_.push_back(std::make_pair(filename, label));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  generate_pair_lines(lines_, pair_lines_);

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color, interpolation, resize_mode);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void SiameseDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());

  vector<PAIR_STRUCT> pair_struct_list;
  creat_pair_struct(pair_lines_, pair_struct_list);
  shuffle(pair_struct_list.begin(), pair_struct_list.end(), prefetch_rng);
  back_to_pair_lines(pair_struct_list, pair_lines_);
}

template <typename Dtype>
void SiameseDataLayer<Dtype>::generate_pair_lines(vector<std::pair<std::string, int> > lines,
											vector<std::pair<std::string, int> > &pair_lines)
{
	int num_samples = static_cast<int>(lines.size());
	int num_classes_ = 0;

	//统计样本中的类别数
	std::vector<int> class_cnt(num_samples, 0);

	for (int i = 0; i < num_samples; i++)
	{
		int label = lines[i].second;
		class_cnt.at(label) = 1;
	}

	for (int i = 0; i < num_samples; i++)
	{
		num_classes_ += class_cnt.at(i);
	}

	//获取每一类的样本
	std::vector<std::vector<int> > sample_list_per_class(num_classes_);
	for (int i = 0; i < num_samples; ++i)
	{
		int label = lines[i].second;
		sample_list_per_class.at(label).push_back(i);
	}

	//生成pair, 每一个样本生成一个正对和一个负对
	int inner_class_sample_num;
	int n1, n2, c2;
	int smp_idx;

	vector<PAIR_SAMPLE>	pairs_lines;
	PAIR_SAMPLE tmp_smp;

	for (int i = 0; i < num_samples; i++)
	{
		int label = lines[i].second;

		inner_class_sample_num = (int)sample_list_per_class.at(label).size();

		//随机生成同一个类别中的不同的样本
		n1 = rand() % inner_class_sample_num;

		//正样本对
		smp_idx = sample_list_per_class.at(label).at(n1);
		tmp_smp.smp_idx = i;
		tmp_smp.smp_label = lines[i].second;
		tmp_smp.pair_idx = smp_idx;
		tmp_smp.pair_label = lines[smp_idx].second;
		tmp_smp.label = 1;
		pairs_lines.push_back(tmp_smp);

		//随机生成一个不同的类别
		c2 = rand() % num_classes_;
		while (c2 == label)
		{
			c2 = rand() % num_classes_;
		}

		//随机生成不同的类别下的id
		inner_class_sample_num = (int)sample_list_per_class.at(c2).size();
		n2 = rand() % inner_class_sample_num;

		//负样本对
		smp_idx = sample_list_per_class.at(c2).at(n2);
		tmp_smp.smp_idx = i;
		tmp_smp.smp_label = lines[i].second;
		tmp_smp.pair_idx = smp_idx;
		tmp_smp.pair_label = lines[smp_idx].second;
		tmp_smp.label = 0;
		pairs_lines.push_back(tmp_smp);
	}

	std::pair<std::string, int>  p1, p2;
	for (int i = 0; i < pairs_lines.size(); i++)
	{
		p1 = lines[pairs_lines[i].smp_idx];
		p2 = lines[pairs_lines[i].pair_idx];
		pair_lines.push_back(p1);
		pair_lines.push_back(p2);
	}
}

template <typename Dtype>
void SiameseDataLayer<Dtype>::creat_pair_struct(vector<std::pair<std::string, int> > pair_lines, 
													vector<PAIR_STRUCT>					&pair_struct_list)
{
	PAIR_STRUCT pair_struct;
	for (int i = 0; i < pair_lines.size() / 2; i++)
	{
		pair_struct.line[0] = pair_lines[i * 2];
		pair_struct.line[1] = pair_lines[i * 2 + 1];
		pair_struct_list.push_back(pair_struct);
	}
}
template <typename Dtype>
void SiameseDataLayer<Dtype>::back_to_pair_lines(vector<PAIR_STRUCT>					pair_struct_list, 
														vector<std::pair<std::string, int> > &pair_lines)
{
	std::pair<std::string, int> pair_line1, pair_line2;
	for (int i = 0; i < pair_struct_list.size(); i++)
	{
		pair_line1 = pair_struct_list[i].line[0];
		pair_line2 = pair_struct_list[i].line[1];
		pair_lines.push_back(pair_line1);
		pair_lines.push_back(pair_line2);
	}
}
// This function is called on prefetch thread
template <typename Dtype>
void SiameseDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();  
  const int interpolation = image_data_param.interpolation();
  const int resize_mode = image_data_param.resize_mode();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + pair_lines_[lines_id_].first,
      new_height, new_width, is_color, interpolation, resize_mode);
  CHECK(cv_img.data) << "Could not load " << pair_lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = pair_lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
	cv::Mat cv_img = ReadImageToCVMat(root_folder + pair_lines_[lines_id_].first,
        new_height, new_width, is_color, interpolation, resize_mode);
	CHECK(cv_img.data) << "Could not load " << pair_lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
	//cv::imwrite("aa.jpg", cv_img);
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

	prefetch_label[item_id] = pair_lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(SiameseDataLayer);
REGISTER_LAYER_CLASS(SiameseData);

}  // namespace caffe
#endif  // USE_OPENCV
