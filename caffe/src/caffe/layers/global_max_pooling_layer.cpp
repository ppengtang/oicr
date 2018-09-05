#include "caffe/layers/global_max_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void GlobalMaxPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  vector<int> top_shape = bottom[0]->shape();
  top_shape[0] = 1;
  top[0]->Reshape(top_shape);

  vector<int> idx_shape = bottom[0]->shape();
  idx_shape[0] = 1;
  max_idx_.Reshape(idx_shape);
}

template <typename Dtype>
void GlobalMaxPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int* real_max_idx = max_idx_.mutable_cpu_data();

  const int outer_num_ = bottom[0]->shape(0);
  const int K_ = bottom[0]->count() / outer_num_;
  
  for (int k = 0; k < K_; k++) {
    top_data[k] = bottom_data[k];
    real_max_idx[k] = 0;
    for (int i = 1; i < outer_num_; i++) {
      if (top_data[k] < bottom_data[i * K_ + k]) {
        top_data[k] = bottom_data[i * K_ + k];
        real_max_idx[k] = i;
      }
    }
  }
}

template <typename Dtype>
void GlobalMaxPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Propagate to bottom
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int* real_max_idx = max_idx_.cpu_data();
  
    const int outer_num_ = bottom[0]->shape(0);
    const int K_ = bottom[0]->count() / outer_num_;

    for (int i = 0; i < outer_num_; i++) {
      for (int k = 0; k < K_; k++) {
        bottom_diff[i * K_ + k] = top_diff[k] * Dtype(real_max_idx[k] == i);
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(GlobalMaxPoolingLayer);
#endif

INSTANTIATE_CLASS(GlobalMaxPoolingLayer);
REGISTER_LAYER_CLASS(GlobalMaxPooling);

}  // namespace caffe
