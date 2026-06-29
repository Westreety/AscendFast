
#include "register/tilingdata_base.h"

namespace optiling {
// RmsNormCustom 的 tiling：按行规约的 RMSNorm。
//   numRows  = 行数（输入归一为 [numRows, hidden] 后的行数）。
//   hidden   = 每行元素数（规约维 = gamma 长度）。
//   epsilon  = 数值稳定项，host 从 attr 读出后传给 device。
// 多核切分按行：device 侧用 GetBlockIdx()/GetBlockNum() 自算本核行范围，
// 因此 tiling 不需要存 rowsPerCore——避免 numRows 不整除时的尾块记账。
BEGIN_TILING_DATA_DEF(RmsNormCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numRows);
  TILING_DATA_FIELD_DEF(uint32_t, hidden);
  TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RmsNormCustom, RmsNormCustomTilingData)
}
