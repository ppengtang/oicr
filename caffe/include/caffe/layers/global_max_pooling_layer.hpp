#ifndef CAFFE_GLOBAL_MAX_POOLING_LAYER_HPP_
#define CAFFE_GLOBAL_MAX_POOLING_LAYER_HPP_

#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
class GlobalMaxPoolingLayer : public Layer<Dtype> {
 public:
  explicit GlobalMaxPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GlobalMaxPooling"; }

  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<int> max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_GLOBAL_MAX_POOLING_LAYER_HPP_
