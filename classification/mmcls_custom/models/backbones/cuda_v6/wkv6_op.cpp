#include <torch/extension.h>
#include "ATen/ATen.h"

void cuda_forward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *y);
void cuda_backward(int B, int T, int C, int H, float *r, float *k, float *v, float *w, float *u, float *gy, float *gr, float *gk, float *gv, float *gw, float *gu);

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    cuda_forward(B, T, C, H, r.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), w.data_ptr<float>(), u.data_ptr<float>(), y.data_ptr<float>());
}
void backward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &gy, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &gw, torch::Tensor &gu) {
    cuda_backward(B, T, C, H, r.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), w.data_ptr<float>(), u.data_ptr<float>(), gy.data_ptr<float>(), gr.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>(), gw.data_ptr<float>(), gu.data_ptr<float>());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv6 forward");
    m.def("backward", &backward, "wkv6 backward");
}

TORCH_LIBRARY(wkv6, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}