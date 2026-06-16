#!/usr/bin/env bash
source scripts/ascend-env.sh
cd /models/share/userdata/cb/AscendFast/adaptations/Qwen2.5-0.5B-Instruct/mode_Qwen2.5-0.5B-Instruct_2_1781508028
python _probe_load.py 2>&1 | tail -30
