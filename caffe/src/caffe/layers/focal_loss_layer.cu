#include "caffe/layers/focal_loss_layer.hpp"

using std::max;

namespace caffe {
template <typename Dtype>
__global__ void FocalLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* label, 
          Dtype gamma, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype prob = input_data[index];
    const Dtype epsilon = 1e-6;
    if (int(label[index]) == 0) {
      loss[index] = -log(max(1 - prob, epsilon));
    }
    else {
      loss[index] = -log(max(prob, epsilon));
    }
  }
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int nthreads = batch_size * channels;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  // NOLINT_NEXT_LINE(whitespace/operators)
  FocalLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, label, gamma_, loss_data);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  loss /= batch_size;
  top[0]->mutable_cpu_data()[0] = loss;
}
template <typename Dtype>
__global__ void FocalLossBackwardGPU(const int nthreads, 
          const Dtype* input_data, const Dtype* label, Dtype gamma,
          Dtype* bottom_diff, const int channels) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype prob = input_data[index];
    const Dtype epsilon = 1e-6;
    if (int(label[index]) == 0) {
      bottom_diff[index] = pow(max(prob, epsilon), gamma) / max(1 - prob, epsilon)
          - gamma * pow(max(prob, epsilon), gamma - 1) * log(max(1 - prob, epsilon));
    }
    else {
      bottom_diff[index] = -pow(max(1 - prob, epsilon), gamma) / max(prob, epsilon) 
          + gamma * pow(max(1 - prob, epsilon), gamma - 1) * log(max(prob, epsilon));
    }
  }
}
template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* input_data = bottom[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int nthreads = batch_size * channels;
    
    FocalLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, label, gamma_, bottom_diff,
        channels);
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    
    caffe_gpu_scal(bottom[0]->count(), loss_weight / batch_size, bottom_diff);
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(FocalLossLayer);
}  // namespace caffe
