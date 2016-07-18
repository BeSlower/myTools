#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#endif  // USE_OPENCV

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#define  M_PI  3.14159265358979323846
namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param,
    Phase phase)
    : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  //const int crop_size = param_.crop_size();
  const int crop_size_height = param_.crop_size_height();
  const int crop_size_width = param_.crop_size_width();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size_height);
  CHECK_GE(datum_width, crop_size_width);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size_height && crop_size_width) {
    height = crop_size_height;
    width = crop_size_width;
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(datum_height - crop_size_height + 1);
      w_off = Rand(datum_width - crop_size_width + 1);
    } else {
      h_off = (datum_height - crop_size_height) / 2;
      w_off = (datum_width - crop_size_width) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  // If datum is encoded, decoded and transform the cv::image.
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // Transform the cv::image into blob.
    return Transform(cv_img, transformed_blob);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  } else {
    if (param_.force_color() || param_.force_gray()) {
      LOG(ERROR) << "force_color and force_gray only for encoded datum";
    }
  }

  //const int crop_size = param_.crop_size();
  const int crop_size_height = param_.crop_size_height();
  const int crop_size_width = param_.crop_size_width();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  // Check dimensions.
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  if (crop_size_height && crop_size_width) {
    CHECK_EQ(crop_size_height, height);
    CHECK_EQ(crop_size_width, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be no greater than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

#ifdef USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::ColorCasting(cv::Mat &img, const int magnitude)
{
	bool is_casting[3];
	for (int i = 0; i < 3; i++)
	{
		is_casting[i] = Rand(2);
	}

	vector<cv::Mat> channels;
	cv::split(img, channels);

	for (int i = 0; i < img.channels(); ++i)
	{
		if (is_casting[i])
		{
			channels.at(i) += (int)Rand(magnitude * 2 + 1) - magnitude;
		}
	}

	cv::merge(channels, img);
}

template<typename Dtype>
float DataTransformer<Dtype>::ResizeImageKeepAspectRatio(cv::Mat &img, 
														const int width, 
														const int height)
{
    int w = img.cols;
    int h = img.rows;
    int w_r, h_r;

    float scale_factor = 1.f;
    float target_ratio = static_cast<float>(width) / static_cast<float>(height);
    float ratio = static_cast<float>(w) / static_cast<float>(h);

    if (ratio <= target_ratio) //w是短边
    {
        w_r = width;
        scale_factor = static_cast<float>(w_r) / static_cast<float>(w);
        h_r = static_cast<int>(static_cast<float>(w_r) / ratio + 0.5f);
    }
    else //h是短边
    {
        h_r = height;
        scale_factor = static_cast<float>(h_r) / static_cast<float>(h);
        w_r = static_cast<int>(static_cast<float>(h_r) * ratio + 0.5f);
    }

    if (w != w_r && h != h_r)
    {
        //cerr << "Resizing from (" << w << " x " << h << ") to (" << w_r << " x " << h_r << ")" << endl;
        cv::resize(img, img, cv::Size(w_r, h_r), 0, 0, cv::INTER_CUBIC);
    }

    return scale_factor;
}

template<typename Dtype>
int DataTransformer<Dtype>::RotateImageRandomly(cv::Mat &img, const int angle_range)
{
	float thr = 0.5f;
	//int angle = (int)gjrand_rand32mod(rng, angle_range * 2 + 1) - angle_range;
	int angle = (int)Rand(angle_range * 2 + 1) - angle_range;

	if (angle == 0)
	{
		return angle;
	}

	double rad = abs(angle / 180.0 * M_PI);
	double scale = 1.0;

	int w = img.cols;
	int h = img.rows;
	int w_thr = static_cast<int>(float(w) * thr);
	int h_thr = static_cast<int>(float(h) * thr);

	double tmp_w = double(w) * cos(rad) - double(h) * sin(rad);
	double tmp_h = double(h) * cos(rad) - double(w) * sin(rad);
	double cos2alpha = cos(2 * rad);

	if (tmp_w > 0 && tmp_h > 0 && cos2alpha > 0)
	{
		int crop_w = static_cast<int>(tmp_w / cos2alpha);
		int crop_h = static_cast<int>(tmp_h / cos2alpha);

		// Make sure that size after rotation is big enough
		if (crop_w <= w_thr || crop_h <= h_thr)
		{
			return 0;
		}

		//cout << "Input: " << w << " x " << h << endl;
		//cout << "Crop: " << crop_w << " x " << crop_h << endl;

		// Get the rotation matrix with the specifications above
		cv::Point center = cv::Point(w / 2, h / 2);
		cv::Mat rot_mat = cv::getRotationMatrix2D(center, static_cast<double>(angle), scale);

		// Rotate the warped image
		cv::Mat rot_img;
		cv::warpAffine(img, rot_img, rot_mat, img.size());

		int rot_w = rot_img.cols;
		int rot_h = rot_img.rows;

		crop_w = min(crop_w, rot_w);
		crop_h = min(crop_h, rot_h);
		cv::Mat patch = rot_img(cv::Rect((rot_w - crop_w) / 2, (rot_h - crop_h) / 2, crop_w, crop_h));

		img = patch.clone();
	}

	return angle;
}
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<cv::Mat> & mat_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int mat_num = mat_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(mat_num, 0) << "There is no MAT to add";
  CHECK_EQ(mat_num, num) <<
    "The size of mat_vector must be equals to transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < mat_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(mat_vector[item_id], &uni_blob);
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
    Blob<Dtype>* transformed_blob) {
    //const int crop_size = param_.crop_size();
    const int crop_size_height = param_.crop_size_height();
    const int crop_size_width = param_.crop_size_width();
    const int img_channels = cv_img.channels();
    const int img_height = cv_img.rows;
    const int img_width = cv_img.cols;

    // Check dimensions.
    const int channels = transformed_blob->channels();
    const int height = transformed_blob->height();
    const int width = transformed_blob->width();
    const int num = transformed_blob->num();

    CHECK_EQ(channels, img_channels);
    CHECK_LE(height, img_height);
    CHECK_LE(width, img_width);
    CHECK_GE(num, 1);

    CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

    const Dtype scale = param_.scale();
    const bool do_mirror = param_.mirror() && Rand(2);
    const bool has_mean_file = param_.has_mean_file();
    const bool has_mean_values = mean_values_.size() > 0;
    const bool has_rand_smooth = param_.has_rand_smooth_scale();
    const bool has_color_cast = param_.has_color_cast();
    const bool has_aspect_ratio = param_.has_aspect_ratio();
    const bool has_rotation = param_.has_rotation();
    const bool has_scale_jittering = param_.has_scale_jittering();

    CHECK_GT(img_channels, 0);
    CHECK_GE(img_height, crop_size_height);
    CHECK_GE(img_width, crop_size_width);

    Dtype* mean = NULL;
    if (has_mean_file) {
        CHECK_EQ(img_channels, data_mean_.channels());
        CHECK_EQ(img_height, data_mean_.height());
        CHECK_EQ(img_width, data_mean_.width());
        mean = data_mean_.mutable_cpu_data();
    }
    if (has_mean_values) {
        CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
            "Specify either 1 mean_value or as many as channels: " << img_channels;
        if (img_channels > 1 && mean_values_.size() == 1) {
            // Replicate the mean_value for simplicity
            for (int c = 1; c < img_channels; ++c) {
                mean_values_.push_back(mean_values_[0]);
            }
        }
    }

    cv::Mat cv_smoothed_img = cv_img;
    if ((phase_ == TRAIN) && (has_rand_smooth))
    {
        const unsigned int max_smooth_scale = param_.rand_smooth_scale();
        const int smooth_size = Rand(max_smooth_scale + 1);
        if (smooth_size > 0)
        {
            cv::GaussianBlur(cv_img, cv_smoothed_img, cv::Size(2 * smooth_size + 1, 2 * smooth_size + 1), 0);
        }
    }

    int h_off = 0;
    int w_off = 0;
    cv::Mat cv_cropped_img = cv_smoothed_img;
    if (phase_ == TRAIN && crop_size_height && crop_size_width)
    {
        // Data augmentation 0: rotation
        ////////////////////////////////
        if (has_rotation > 0)
        {
            const int rotation = param_.rotation();
            //printf("%s before rotation: %d x %d\n", train_samples_.at(sample_idx).img_name.c_str(), img.cols, img.rows);
            int angle = RotateImageRandomly(cv_cropped_img, rotation);

            // 	  if (img.size().area() <= 0)
            // 	  {
            // 		  fprintf(stderr, "%s after rotation: %d x %d, angle: %d\n",
            // 			  train_samples_.at(sample_idx).img_name.c_str(), img.cols, img.rows, angle);
            // 	  }
        }
        // Data augmentation 1: scale jittering
        ///////////////////////////////////////
        int rand_img_width = img_width;
        int rand_img_height = img_height;
        if (has_scale_jittering)
        {
            //长宽均放大，并保证大于原先的输入尺寸，如256*256
            float scale_jittering = param_.scale_jittering();
            if (scale_jittering > 1.0)
            {
                float rand_scale = 1.0 + (scale_jittering - 1.0) * (float)Rand(100) / 100;
                rand_img_width = static_cast<int>(static_cast<float>(img_width)* rand_scale);
                rand_img_height = static_cast<int>(static_cast<float>(img_height)* rand_scale);
            }
        }
        cv::resize(cv_cropped_img, cv_cropped_img, cv::Size(rand_img_width, rand_img_height), 0, 0, cv::INTER_CUBIC);
        // Data augmentation 2: aspect ratio distortion
        ///////////////////////////////////////////////
        int rand_crop_width = crop_size_width;
        int rand_crop_height = crop_size_height;
        if (has_aspect_ratio)
        {
            //长宽比是针对裁剪尺寸而言，将原有裁剪尺寸的长或者宽缩小一定倍数
            float aspect_ratio = param_.aspect_ratio();
            if (aspect_ratio > 1.0)
            {
                float rand_aspect_ratio = 1.0 + (aspect_ratio - 1.0) * (float)Rand(100) / 100;
                bool short_dim_is_width = (bool)Rand(1);

                if (short_dim_is_width)
                {
                    rand_crop_width = static_cast<int>(static_cast<float>(crop_size_width) / rand_aspect_ratio);
                }
                else
                {
                    rand_crop_height = static_cast<int>(static_cast<float>(crop_size_height) / rand_aspect_ratio);
                }
            }
        }
        // Data augmentation 3: random cropping
        ///////////////////////////////////////
        int w = cv_cropped_img.cols;
        int h = cv_cropped_img.rows;

        //随机选择一个位置裁剪
        int x_off = Rand(w - rand_crop_width + 1);
        int y_off = Rand(h - rand_crop_height + 1);

        cv::Mat patch = cv_cropped_img(cv::Rect(x_off, y_off, rand_crop_width, rand_crop_height));

        if (has_aspect_ratio && (param_.aspect_ratio() > 1.0))
        {
            //至此保证了crop_size * crop_size的大小
            cv::resize(patch, cv_cropped_img, cv::Size(crop_size_width, crop_size_height), 0, 0, cv::INTER_CUBIC);
        }
        else
        {
            cv_cropped_img = patch;
        }
        // Data augmentation 4: Baidu's color casting
        /////////////////////////////////////////////
        if (has_color_cast)
        {
            int color_cast = param_.color_cast();
            ColorCasting(cv_cropped_img, color_cast);
        }
        w_off = (float)x_off / w * img_width;
        h_off = (float)y_off / h * img_height;
    }
    else
    {
        // test data
        if (crop_size_width && crop_size_height)
        {
            CHECK_EQ(crop_size_height, height);
            CHECK_EQ(crop_size_width, width);
            h_off = (img_height - crop_size_height) / 2;
            w_off = (img_width - crop_size_width) / 2;
            cv::Rect roi(w_off, h_off, crop_size_width, crop_size_height);
            cv_cropped_img = cv_smoothed_img(roi);
        }
    }
    //    cv::imwrite("bb.jpg", cv_cropped_img);
    //    cv::waitKey(0);
    CHECK(cv_cropped_img.data);
//     float *bgr = (float *)malloc(height * width * img_channels * sizeof(float));
//     float *b = bgr;
//     float *g = bgr + height * width;
//     float *r = g + height * width;

    Dtype* transformed_data = transformed_blob->mutable_cpu_data();
    int top_index;
    for (int h = 0; h < height; ++h) {
        const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < img_channels; ++c) {
                if (do_mirror) {
                    top_index = (c * height + h) * width + (width - 1 - w);
                }
                else {
                    top_index = (c * height + h) * width + w;
                }
                // int top_index = (c * height + h) * width + w;
                Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
                if (has_mean_file) {
                    int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
                    transformed_data[top_index] =
                        (pixel - mean[mean_index]) * scale;
                }
                else {
                    if (has_mean_values) {
                        transformed_data[top_index] =
                            (pixel - mean_values_[c]) * scale;
                    }
                    else {
                        transformed_data[top_index] = pixel * scale;
                    }
                }
//                 if (img_index % 3 == 0)
//                 {
//                     *b = pixel;
//                     b++;
//                 }
//                 else if (img_index % 3 == 1)
//                 {
//                     *g = pixel;
//                     g++;
//                 }
//                 else
//                 {
//                     *r = pixel;
//                     r++;
//                 }
            }
        }
    }
    //save b,g,r
//     FILE *fp_b = fopen("b.txt", "wt");
//     FILE *fp_g = fopen("g.txt", "wt");
//     FILE *fp_r = fopen("r.txt", "wt");
//     b = bgr;
//     g = bgr + height * width;
//     r = g + height * width;
//     for (int i = 0; i < height * width; i++)
//     {
//         fprintf(fp_b, "%.0f ", b[i]);
//         fprintf(fp_g, "%.0f ", g[i]);
//         fprintf(fp_r, "%.0f ", r[i]);
//     }
//     fclose(fp_b);
//     fclose(fp_g);
//     fclose(fp_r);
//     free(bgr);
    for (int kk = 0; kk < width * height * 3; kk++)
    {
        transformed_data[kk] = 100;
    }
    //memset(transformed_data, (Dtype)100, width * height * 3);
    cout << "aa" << endl;
}
#endif  // USE_OPENCV

template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  //const int crop_size = param_.crop_size();
  const int crop_size_width = param_.crop_size_width();
  const int crop_size_height = param_.crop_size_height();
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
	  if (crop_size_width && crop_size_height) {
      transformed_blob->Reshape(input_num, input_channels,
                                crop_size_height, crop_size_width);
    } else {
      transformed_blob->Reshape(input_num, input_channels,
                                input_height, input_width);
    }
  }

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size_width && crop_size_height) {
    CHECK_EQ(crop_size_height, height);
    CHECK_EQ(crop_size_width, width);
    // We only do random crop when we do training.
    if (phase_ == TRAIN) {
      h_off = Rand(input_height - crop_size_height + 1);
      w_off = Rand(input_width - crop_size_width + 1);
    } else {
      h_off = (input_height - crop_size_height) / 2;
      w_off = (input_width - crop_size_width) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const Datum& datum) {
  if (datum.encoded()) {
#ifdef USE_OPENCV
    CHECK(!(param_.force_color() && param_.force_gray()))
        << "cannot set both force_color and force_gray";
    cv::Mat cv_img;
    if (param_.force_color() || param_.force_gray()) {
    // If force_color then decode in color otherwise decode in gray.
      cv_img = DecodeDatumToCVMat(datum, param_.force_color());
    } else {
      cv_img = DecodeDatumToCVMatNative(datum);
    }
    // InferBlobShape using the cv::image.
    return InferBlobShape(cv_img);
#else
    LOG(FATAL) << "Encoded datum requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  }
  //const int crop_size = param_.crop_size();
  const int crop_size_height = param_.crop_size_height();
  const int crop_size_width = param_.crop_size_width();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();
  // Check dimensions.
  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size_height);
  CHECK_GE(datum_width, crop_size_width);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = datum_channels;
  shape[2] = (crop_size_height)? crop_size_height: datum_height;
  shape[3] = (crop_size_width)? crop_size_width: datum_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<Datum> & datum_vector) {
  const int num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to in the vector";
  // Use first datum in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(datum_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}

#ifdef USE_OPENCV
template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(const cv::Mat& cv_img) {
  //const int crop_size = param_.crop_size();
  const int crop_size_height = param_.crop_size_height();
  const int crop_size_width = param_.crop_size_width();
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;
  // Check dimensions.
  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size_height);
  CHECK_GE(img_width, crop_size_width);
  // Build BlobShape.
  vector<int> shape(4);
  shape[0] = 1;
  shape[1] = img_channels;
  shape[2] = (crop_size_height) ? crop_size_height : img_height;
  shape[3] = (crop_size_width) ? crop_size_width : img_width;
  return shape;
}

template<typename Dtype>
vector<int> DataTransformer<Dtype>::InferBlobShape(
    const vector<cv::Mat> & mat_vector) {
  const int num = mat_vector.size();
  CHECK_GT(num, 0) << "There is no cv_img to in the vector";
  // Use first cv_img in the vector to InferBlobShape.
  vector<int> shape = InferBlobShape(mat_vector[0]);
  // Adjust num to the size of the vector.
  shape[0] = num;
  return shape;
}
#endif  // USE_OPENCV

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size_height() && param_.crop_size_width());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
