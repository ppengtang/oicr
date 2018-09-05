#include "caffe/layers/global_sum_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GlobalSumPoolingForwardGPU(const int nthreads,
          const Dtype* bottom_data, Dtype* top_data,
          const int outer_num_, const int K_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    top_data[index] = 0;
    for (int i = 0; i < outer_num_; i++) {
      top_data[index] += bottom_data[i * K_ + index];
    }
  }
}

template <typename Dtype>
void GlobalSumPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int outer_num_ = bottom[0]->shape(0);
  const int K_ = bottom[0]->count() / outer_num_;

  const int nthreads = K_;
  GlobalSumPoolingForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, top_data, outer_num_, 
                                K_);
}

template <typename Dtype>
__global__ void GlobalSumPoolingBackwardGPU(const int nthreads,
          const Dtype* top_diff, Dtype* bottom_diff, const int K_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int k = index % K_;
    bottom_diff[index] = top_diff[k];
  }
}

template <typename Dtype>
void GlobalSumPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  
    const int outer_num_ = bottom[0]->shape(0);
    const int K_ = bottom[0]->count() / outer_num_;
    const int nthreads = outer_num_ * K_;
    GlobalSumPoolingBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, bottom_diff, K_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GlobalSumPoolingLayer);
}  // namespace caffe
