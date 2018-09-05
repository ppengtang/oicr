#include "caffe/layers/focal_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void FocalLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  gamma_ = this->layer_param_.focal_loss_param().gamma();
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  batch_size = bottom[0]->num();
  channels = bottom[0]->channels();
  CHECK_EQ(batch_size, bottom[1]->num())
  	  << "Number of labels must match number of predictions.";
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype epsilon = 1e-9;
  Dtype loss = 0;
  for (int i = 0; i < batch_size; i++) {
  	for (int c = 0; c < channels; c++) {
      Dtype prob = input_data[i * channels + c];
      if (label[i * channels + c] == 1) {
        loss -= pow(1 - prob, gamma_) * log(std::max(prob, epsilon));
      }
      else {
        loss -= pow(prob, gamma_) * log(std::max(1 - prob, epsilon));
      }
  	}
  }
  top[0]->mutable_cpu_data()[0] = loss / batch_size;
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
  	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype epsilon = 1e-9;
    
    for (int i = 0; i < batch_size; ++i) {
      for (int c = 0; c < channels; c++) {
        Dtype prob = input_data[i * channels + c];
        if (label[i * channels + c] == 1) {
          bottom_diff[i * channels + c] = -pow(1 - prob, gamma_) / std::max(prob, epsilon) 
              + gamma_ * pow(1 - prob, gamma_ - 1) * log(std::max(prob, epsilon));
        }
        else {
          bottom_diff[i * channels + c] = pow(prob, gamma_) / std::max(1 - prob, epsilon)
              - gamma_ * pow(prob, gamma_ - 1) * log(std::max(1 - prob, epsilon));
        }
      }
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(bottom[0]->count(), loss_weight / batch_size, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(FocalLossLayer);
#endif

INSTANTIATE_CLASS(FocalLossLayer);
REGISTER_LAYER_CLASS(FocalLoss);
}
