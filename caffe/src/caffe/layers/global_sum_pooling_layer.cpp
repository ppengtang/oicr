#include "caffe/layers/global_sum_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void GlobalSumPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  vector<int> top_shape = bottom[0]->shape();
  top_shape[0] = 1;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void GlobalSumPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int outer_num_ = bottom[0]->shape(0);
  const int K_ = bottom[0]->count() / outer_num_;
  
  for (int k = 0; k < K_; k++) {
    top_data[k] = 0;
    for (int i = 0; i < outer_num_; i++) {
      top_data[k] += bottom_data[i * K_ + k];
    }
  }
}

template <typename Dtype>
void GlobalSumPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Propagate to bottom
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
    const int outer_num_ = bottom[0]->shape(0);
    const int K_ = bottom[0]->count() / outer_num_;

    for (int i = 0; i < outer_num_; i++) {
      for (int k = 0; k < K_; k++) {
        bottom_diff[i * K_ + k] = top_diff[k];
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(GlobalSumPoolingLayer);
#endif

INSTANTIATE_CLASS(GlobalSumPoolingLayer);
REGISTER_LAYER_CLASS(GlobalSumPooling);

}  // namespace caffe
