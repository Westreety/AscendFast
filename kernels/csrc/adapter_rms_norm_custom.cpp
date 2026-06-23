// rms_norm 的 PyTorch 适配层（路线 A：手写 aclnn 两段式），仿 adapter_add_demo.cpp。
//
// 注册算子 torch.ops.ascendfast.rms_norm(x, gamma, eps) -> y，语义等价于
//   y = x * rsqrt(mean(x^2, dim=-1) + eps) * gamma
// device kernel 把输入按「最后一维 = hidden、其余 = 行」做按行规约，所以 adapter
// 直接传连续的原始张量即可（无需 reshape）；out 与 x 同形。
//
// 关键：命名空间 ascendfast 的 TORCH_LIBRARY 已在 adapter_add_demo.so 里定义过一次。
// 本 .so 若再用 TORCH_LIBRARY(ascendfast,...) 会和它撞「命名空间重复定义」。所以这里
// 用 TORCH_LIBRARY_FRAGMENT —— 它允许多个 .so 往同一个已存在命名空间里**追加** schema。
#include <vector>

#include <torch/extension.h>

#include "acl/acl.h"
#include "aclnn_rms_norm_custom.h"

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h"

namespace ascendfast {

// at::Tensor -> aclTensor。RMSNorm 走 ND、连续，view/storage 同形，行主序 stride。
static aclTensor* to_acl_rms(const at::Tensor& t, aclDataType dtype) {
    std::vector<int64_t> dims(t.sizes().begin(), t.sizes().end());
    std::vector<int64_t> strides(t.strides().begin(), t.strides().end());
    return aclCreateTensor(
        dims.data(), dims.size(), dtype,
        strides.data(), /*offset=*/0, ACL_FORMAT_ND,
        dims.data(), dims.size(), t.data_ptr());
}

static aclDataType acl_dtype_of_rms(const at::Tensor& t) {
    switch (t.scalar_type()) {
        case at::kHalf:  return ACL_FLOAT16;
        case at::kFloat: return ACL_FLOAT;
        default:
            TORCH_CHECK(false, "rms_norm: unsupported dtype ", t.scalar_type(),
                        " (only float16/float32)");
    }
}

at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& gamma, double eps) {
    TORCH_CHECK(x.scalar_type() == gamma.scalar_type(),
                "rms_norm: x/gamma dtype mismatch");
    TORCH_CHECK(gamma.dim() == 1, "rms_norm: gamma must be 1-D [hidden]");
    TORCH_CHECK(x.size(-1) == gamma.size(0),
                "rms_norm: x last dim must equal gamma length");

    at::Tensor xc = x.contiguous();
    at::Tensor gc = gamma.contiguous();
    at::Tensor out = at::empty_like(xc);

    aclDataType dt = acl_dtype_of_rms(xc);
    aclTensor* ax = to_acl_rms(xc, dt);
    aclTensor* ag = to_acl_rms(gc, dt);
    aclTensor* ay = to_acl_rms(out, dt);

    aclrtStream stream = c10_npu::getCurrentNPUStream();

    // 第一段：问工作区大小 + 拿 executor。eps 作为 double 传入（aclnn 自动生成的签名）。
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus ret = aclnnRmsNormCustomGetWorkspaceSize(
        ax, ag, eps, ay, &wsSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnRmsNormCustomGetWorkspaceSize failed: ", ret);

    at::Tensor ws;
    void* wsPtr = nullptr;
    if (wsSize > 0) {
        ws = at_npu::native::allocate_workspace(wsSize, stream);
        wsPtr = ws.data_ptr();
    }

    // 第二段：真正下发到 device。
    ret = aclnnRmsNormCustom(wsPtr, wsSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnRmsNormCustom failed: ", ret);

    aclDestroyTensor(ax);
    aclDestroyTensor(ag);
    aclDestroyTensor(ay);
    return out;
}

}  // namespace ascendfast

// FRAGMENT：往已存在的 ascendfast 命名空间追加 schema（不与 add_demo 的 TORCH_LIBRARY 冲突）。
TORCH_LIBRARY_FRAGMENT(ascendfast, m) {
    m.def("rms_norm(Tensor x, Tensor gamma, float eps) -> Tensor");
}

// NPU 走 PrivateUse1 dispatch key。
TORCH_LIBRARY_IMPL(ascendfast, PrivateUse1, m) {
    m.impl("rms_norm", &ascendfast::rms_norm);
}
