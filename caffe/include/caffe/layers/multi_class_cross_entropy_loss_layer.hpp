#ifndef CAFFE_MULTI_CLASS_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_MULTI_CLASS_CROSS_ENTROPY_LOSS_LAYER_HPP_
#include <algorithm>
#include <cfloat>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
template <typename Dtype>
class MulticlassCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  explicit MulticlassCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "MulticlassCrossEntropyLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int batch_size, channels;
};
}  // namespace caffe
#endif  // CAFFE_MULTI_CLASS_CROSS_ENTROPY_LOSS_LAYER_HPP_