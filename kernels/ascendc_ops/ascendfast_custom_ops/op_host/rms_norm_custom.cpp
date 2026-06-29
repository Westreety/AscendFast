
#include "rms_norm_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
// RMSNorm 按行规约：把输入看成 [numRows, hidden]，每行独立归一化。
// 切分按行交给 device（GetBlockNum/GetBlockIdx 自算范围），host 只需算出
// numRows / hidden / epsilon 三个量，并选定 blockDim（启动多少个核）。
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  RmsNormCustomTilingData tiling;

  const gert::StorageShape* x_shape = context->GetInputShape(0);
  const auto& shp = x_shape->GetStorageShape();
  size_t dimNum = shp.GetDimNum();

  // hidden = 最后一维（规约维，= gamma 长度）；numRows = 其余维之积。
  int64_t hidden = (dimNum >= 1) ? shp.GetDim(dimNum - 1) : 1;
  int64_t numRows = 1;
  for (size_t i = 0; i + 1 < dimNum; i++) {
    numRows *= shp.GetDim(i);
  }
  if (hidden <= 0) hidden = 1;
  if (numRows <= 0) numRows = 1;

  // epsilon：从 attr[0] 读；缺省（attr 未传/为空指针）回落 1e-6。
  float epsilon = 1e-6f;
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  if (attrs != nullptr && attrs->GetAttrNum() > 0) {
    const float* epsPtr = attrs->GetFloat(0);
    if (epsPtr != nullptr) epsilon = *epsPtr;
  }

  // blockDim：按行起核，最多不超过 numRows（避免比行还多的核全空转）。
  // 取一个适中的核数上限即可；device 端对“分到 0 行”的核做空跳处理。
  uint32_t maxCore = 48;
  uint32_t blockDim = static_cast<uint32_t>(numRows < maxCore ? numRows : maxCore);
  if (blockDim < 1) blockDim = 1;

  tiling.set_numRows(static_cast<uint32_t>(numRows));
  tiling.set_hidden(static_cast<uint32_t>(hidden));
  tiling.set_epsilon(epsilon);

  context->SetBlockDim(blockDim);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
  const gert::Shape* x_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x_shape;
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class RmsNormCustom : public OpDef {
public:
    explicit RmsNormCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // epsilon 作为可选 float attr，默认 1e-6，与 Qwen2 rms_norm_eps 一致。
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(RmsNormCustom);
}
