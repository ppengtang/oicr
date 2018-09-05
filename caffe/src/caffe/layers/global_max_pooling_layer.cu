#include "caffe/layers/global_max_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GlobalMaxPoolingForwardGPU(const int nthreads,
          const Dtype* bottom_data, Dtype* top_data, int* real_max_idx,
          const int outer_num_, const int K_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] = bottom_data[index];
    real_max_idx[index] = 0;
    for (int i = 1; i < outer_num_; i++) {
      if (top_data[index] < bottom_data[i * K_ + index]) {
        top_data[index] = bottom_data[i * K_ + index];
        real_max_idx[index] = i;
      }
    }
  }
}

template <typename Dtype>
void GlobalMaxPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* real_max_idx = max_idx_.mutable_gpu_data();

  const int outer_num_ = bottom[0]->shape(0);
  const int K_ = bottom[0]->count() / outer_num_;

  const int nthreads = K_;
  GlobalMaxPoolingForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, top_data, real_max_idx,
                                outer_num_, K_);
}

template <typename Dtype>
__global__ void GlobalMaxPoolingBackwardGPU(const int nthreads,
          const Dtype* top_diff, Dtype* bottom_diff, 
          const int* real_max_idx,const int K_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int k = index % K_;
    const int i = index / K_;
    bottom_diff[index] = top_diff[k] * Dtype(real_max_idx[k] == i);
  }
}

template <typename Dtype>
void GlobalMaxPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int* real_max_idx = max_idx_.gpu_data();
  
    const int outer_num_ = bottom[0]->shape(0);
    const int K_ = bottom[0]->count() / outer_num_;
    const int nthreads = outer_num_ * K_;
    GlobalMaxPoolingBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, bottom_diff, real_max_idx, K_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GlobalMaxPoolingLayer);
}  // namespace caffe
