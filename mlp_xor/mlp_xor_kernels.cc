#include "../../kernels/cpu_kernels.h"
#include "tfrt/cpu/ops/test/cpu_ops_and_kernels.h"

using ::tfrt::compat::NullaryEigenKernelAsync;
using ::tfrt::compat::UnaryEigenKernelAsync;

namespace tfrt {

// Computes A = Sigmoid(A).
static AsyncValueRef<Chain> SigmoidInPlace(DenseHostTensor* A,
                                        const ExecutionContext& exec_ctx) {
  auto fn = [](auto& a) { return 1.0 / (1.0 + (-a).exp()); };
  return NullaryEigenKernelAsync<float>(A, std::move(fn), exec_ctx);
}

static AsyncValueRef<Chain> SigmoidGradInplace(const DenseHostTensor& activation,
                                               DenseHostTensor* gradient,
                                               const ExecutionContext& exec_ctx) {
  auto fn = [](const auto& a, auto& ) {
      return a - a.square();
  };
  return UnaryEigenKernelAsync<float, float>(
          activation, gradient, std::move(fn), exec_ctx);
}

void RegisterMlpXorKernels(KernelRegistry* registry) {
  registry->AddKernel("sigmoid_inplace.f32", TFRT_KERNEL(SigmoidInPlace));
  registry->AddKernel("sigmoid_grad_inplace.f32", TFRT_KERNEL(SigmoidGradInplace));
}

}  // namespace tfrt
