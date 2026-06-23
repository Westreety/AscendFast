#include "kernel_operator.h"

// device 侧 kernel：按行 RMSNorm。把输入看成 [numRows, hidden]，每行独立：
//   y = x * rsqrt(mean(x^2) + eps) * gamma
// 多核切分按行（每核一段连续行，行内规约不跨核）。内部统一用 fp32 计算保精度，
// fp16 输入在 CopyIn 后 Cast 到 fp32、算完再 Cast 回 fp16。
constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class KernelRmsNormCustom {
  // 编译期判定：T 是否为 half（fp16）。fp32 时 sizeof(T)==4，跳过 Cast。
  static constexpr bool isHalf = (sizeof(T) == 2);

public:
  __aicore__ inline KernelRmsNormCustom() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                              uint32_t numRows, uint32_t hidden, float epsilon) {
    this->hidden = hidden;
    this->eps = epsilon;
    // aicore 不允许 float 与 unsigned 直接互转，先经 int32 再转 float。
    this->invHidden = 1.0f / static_cast<float>(static_cast<int32_t>(hidden));

    // 本核负责 [startRow, startRow+rowsThisCore) 这段连续行。
    uint32_t blockNum = AscendC::GetBlockNum();
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t base = numRows / blockNum;
    uint32_t rem = numRows % blockNum;
    // 前 rem 个核各多分 1 行，保证总行数对得上（尾块处理）。
    uint32_t startRow = blockIdx * base + (blockIdx < rem ? blockIdx : rem);
    this->rowsThisCore = base + (blockIdx < rem ? 1 : 0);

    xGm.SetGlobalBuffer((__gm__ T*)x + (uint64_t)startRow * hidden,
                        (uint64_t)this->rowsThisCore * hidden);
    yGm.SetGlobalBuffer((__gm__ T*)y + (uint64_t)startRow * hidden,
                        (uint64_t)this->rowsThisCore * hidden);
    gammaGm.SetGlobalBuffer((__gm__ T*)gamma, hidden);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, hidden * sizeof(T));
    pipe.InitBuffer(inQueueG, 1, hidden * sizeof(T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, hidden * sizeof(T));
    // fp32 计算暂存：xf（归一化工作区）、gf（gamma 常驻）、sq（平方）、work（规约暂存）。
    pipe.InitBuffer(xfBuf, hidden * sizeof(float));
    pipe.InitBuffer(gfBuf, hidden * sizeof(float));
    pipe.InitBuffer(sqBuf, hidden * sizeof(float));
    pipe.InitBuffer(workBuf, hidden * sizeof(float));
    pipe.InitBuffer(redBuf, 32);  // 规约结果（1 个 float），按 32B 对齐分配。
  }

  __aicore__ inline void Process() {
    LoadGamma();  // gamma 行不变，整核只搬一次、转一次 fp32。
    for (uint32_t r = 0; r < this->rowsThisCore; r++) {
      CopyIn(r);
      Compute();
      CopyOut(r);
    }
  }

private:
  __aicore__ inline void LoadGamma() {
    AscendC::LocalTensor<T> gLocal = inQueueG.AllocTensor<T>();
    AscendC::DataCopy(gLocal, gammaGm, this->hidden);
    inQueueG.EnQue(gLocal);
    AscendC::LocalTensor<T> g2 = inQueueG.DeQue<T>();
    AscendC::LocalTensor<float> gf = gfBuf.Get<float>();
    if constexpr (isHalf) {
      AscendC::Cast(gf, g2, AscendC::RoundMode::CAST_NONE, this->hidden);
    } else {
      AscendC::DataCopy(gf, g2, this->hidden);
    }
    inQueueG.FreeTensor(g2);
  }

  __aicore__ inline void CopyIn(uint32_t r) {
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    AscendC::DataCopy(xLocal, xGm[(uint64_t)r * this->hidden], this->hidden);
    inQueueX.EnQue(xLocal);
  }

  __aicore__ inline void Compute() {
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<float> xf = xfBuf.Get<float>();
    if constexpr (isHalf) {
      AscendC::Cast(xf, xLocal, AscendC::RoundMode::CAST_NONE, this->hidden);
    } else {
      AscendC::DataCopy(xf, xLocal, this->hidden);
    }
    inQueueX.FreeTensor(xLocal);

    // sum(x^2) 沿 hidden 规约 → red[0]。
    AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
    AscendC::Mul(sq, xf, xf, this->hidden);
    AscendC::LocalTensor<float> red = redBuf.Get<float>();
    AscendC::LocalTensor<float> work = workBuf.Get<float>();
    AscendC::ReduceSum<float>(red, sq, work, this->hidden);

    // rstd = rsqrt(mean + eps)，在 1 元素上用矢量指令算，再读回标量。
    AscendC::Muls(red, red, this->invHidden, 1);  // mean = sum / hidden
    AscendC::Adds(red, red, this->eps, 1);        // mean + eps
    AscendC::Rsqrt(red, red, 1);                  // rstd
    float rstd = red.GetValue(0);

    // y = (x * rstd) * gamma，仍在 fp32 上做。
    AscendC::LocalTensor<float> gf = gfBuf.Get<float>();
    AscendC::Muls(xf, xf, rstd, this->hidden);
    AscendC::Mul(xf, xf, gf, this->hidden);

    AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    if constexpr (isHalf) {
      AscendC::Cast(yLocal, xf, AscendC::RoundMode::CAST_RINT, this->hidden);
    } else {
      AscendC::DataCopy(yLocal, xf, this->hidden);
    }
    outQueueY.EnQue<T>(yLocal);
  }

  __aicore__ inline void CopyOut(uint32_t r) {
    AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    AscendC::DataCopy(yGm[(uint64_t)r * this->hidden], yLocal, this->hidden);
    outQueueY.FreeTensor(yLocal);
  }

private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueG;
  AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
  AscendC::TBuf<AscendC::TPosition::VECCALC> xfBuf, gfBuf, sqBuf, workBuf, redBuf;
  AscendC::GlobalTensor<T> xGm, yGm, gammaGm;
  uint32_t hidden;
  uint32_t rowsThisCore;
  float eps;
  float invHidden;
};

extern "C" __global__ __aicore__ void rms_norm_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                                      GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // DTYPE_X 由框架按 IR dtype 注入（fp16/fp32 各编一份）。
  KernelRmsNormCustom<DTYPE_X> op;
  op.Init(x, gamma, y, tiling_data.numRows, tiling_data.hidden, tiling_data.epsilon);
  op.Process();
}
