#include "caffe/layers/refine_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void RefineLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  batch_size = bottom[0]->num();
  channels = bottom[0]->channels();
  num_positive = bottom[4]->count();
  CHECK_EQ(batch_size, bottom[1]->count())
      << "Number of labels must match number of predictions.";
  CHECK_EQ(batch_size, bottom[2]->count())
      << "Number of labels must match number of loss weights.";
  CHECK_EQ(batch_size, bottom[3]->count())
      << "Number of labels must match number of gt_assignment.";
  CHECK_EQ(bottom[4]->count(), bottom[5]->count())
      << "bottom[4]->count() must equal to bottom[5]->count().";
  CHECK_EQ(bottom[4]->count(), bottom[6]->count())
      << "bottom[4]->count() must equal to bottom[6]->count().";
  CHECK_EQ(bottom[4]->count(), bottom[7]->count())
      << "bottom[4]->count() must equal to bottom[7]->count().";
  CHECK_EQ(channels, bottom[8]->count())
      << "Number of channels must match number of image labels.";
}

template <typename Dtype>
void RefineLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void RefineLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}
#ifdef CPU_ONLY
STUB_GPU(RefineLossLayer);
#endif
INSTANTIATE_CLASS(RefineLossLayer);
REGISTER_LAYER_CLASS(RefineLoss);
}
