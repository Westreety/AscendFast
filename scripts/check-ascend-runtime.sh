#!/usr/bin/env bash
# 跑前自检：确认项目 venv 能看见 torch + torch_npu，且 NPU 真的可用、能算。

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${script_dir}/ascend-env.sh"

"${VIRTUAL_ENV}/bin/python" - <<'PY'
import os
import sys

import torch
import torch_npu

print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"torch_npu={torch_npu.__version__}")
print(f"transformers={__import__('transformers').__version__}")
print(f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH')}")
print(f"ASCEND_OPP_PATH={os.environ.get('ASCEND_OPP_PATH')}")
print(f"npu_available={torch.npu.is_available()}")
print(f"npu_count={torch.npu.device_count()}")

if not torch.npu.is_available():
    raise SystemExit("torch.npu is not available")

torch.npu.set_device(0)
x = torch.ones((2, 2), device="npu")
y = x + 1
torch.npu.synchronize()
print(f"npu_tensor_sum={float(y.cpu().sum())}")
print("OK: NPU runtime is usable")
PY
