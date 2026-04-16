#!/bin/bash
# =============================================================================
# DC-SatMVS Prediction Script
# =============================================================================
# 通过修改 USE_TQDM 变量来切换 tqdm 进度条和纯日志输出
# USE_TQDM=true  : 使用 tqdm 交互式进度条
# USE_TQDM=false : 使用纯日志打印输出
# =============================================================================

# ==================== 配置参数 ====================
# 进度显示模式 (true: tqdm进度条, false: 纯日志)
USE_TQDM=false

# 基础配置
MODEL="SAMsat"                    # 模型类型: samsat, red, casmvs, ucs
GEO_MODEL="rpc"                   # 几何模型: rpc, pinhole
GPU_ID="0"                        # GPU ID

# 数据配置
DATASET_ROOT="/home/murph_dl/Paper_Re/SatMVS_Re/data/whu_tlc"  # 数据集路径

# 模型检查点
LOAD_CKPT="./checkpoints/samsat/rpc/model_000012.ckpt"  # 模型检查点路径

# 网络参数
VIEW_NUM=3                        # 视角数量
REF_VIEW=2                        # 参考视角
BATCH_SIZE=1                      # 批大小

# 级联参数
NDEPTHS="64,32,8"                 # 各阶段深度数
MIN_INTERVAL=2.5                  # 最小间隔
DEPTH_INTER_R="4,2,1"             # 深度间隔比率
CR_BASE_CHS="8,8,8"               # 代价正则化基础通道数
LAMB=1.5                          # UCS-Net lambda参数

# TensorBoard 参数
SUMMARY_FREQ_SAMPLES=50                   # TensorBoard 日志记录频率（每多少样本记录一次）

# ==================== 构建命令 ====================
CMD="python predict.py \
    --model ${MODEL} \
    --geo_model ${GEO_MODEL} \
    --dataset_root ${DATASET_ROOT} \
    --loadckpt ${LOAD_CKPT} \
    --view_num ${VIEW_NUM} \
    --ref_view ${REF_VIEW} \
    --batch_size ${BATCH_SIZE} \
    --ndepths ${NDEPTHS} \
    --min_interval ${MIN_INTERVAL} \
    --depth_inter_r ${DEPTH_INTER_R} \
    --cr_base_chs ${CR_BASE_CHS} \
    --lamb ${LAMB} \
    --summary_freq_samples ${SUMMARY_FREQ_SAMPLES} \
    --gpu_id ${GPU_ID}"

# 添加 tqdm 参数
if [ "$USE_TQDM" = true ]; then
    CMD="${CMD} --use_tqdm"
    echo "========== 使用 tqdm 进度条模式 =========="
else
    echo "========== 使用纯日志输出模式 =========="
fi

# ==================== 执行预测 ====================
echo "============================================"
echo "开始预测: ${MODEL} 模型, ${GEO_MODEL} 几何模型"
echo "GPU: ${GPU_ID}, Batch Size: ${BATCH_SIZE}"
echo "数据集: ${DATASET_ROOT}"
echo "检查点: ${LOAD_CKPT}"
echo "============================================"
echo ""

eval ${CMD}
