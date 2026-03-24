#!/bin/bash
# =============================================================================
# DC-SatMVS Training Script (TLC Dataset)
# =============================================================================
# 通过修改 USE_TQDM 变量来切换 tqdm 进度条和纯日志输出
# USE_TQDM=true  : 使用 tqdm 交互式进度条
# USE_TQDM=false : 使用纯日志打印输出
# =============================================================================

# ==================== 配置参数 ====================
# 进度显示模式 (true: tqdm进度条, false: 纯日志)
USE_TQDM=false

# 基础配置
MODEL="SAMsat"                    # 模型类型: SAMsat, red, casmvs, ucs
GEO_MODEL="rpc"                   # 几何模型: rpc, pinhole
GPU_ID="0"                        # GPU ID

# 数据配置
DATASET_ROOT="/home/murph_dl/Paper_Re/SatMVS_Re/data/whu_tlc"  # 数据集根目录
LOG_DIR="./checkpoints"           # 日志和检查点保存目录

# 训练超参数
BATCH_SIZE=1                      # 批大小
EPOCHS=15                         # 训练轮数
LR=0.001                          # 学习率
SEED=1                            # 随机种子

# 网络架构参数
VIEW_NUM=3                        # 视角数量
REF_VIEW=2                        # 参考视角
NDEPTHS="64,32,8"                 # 各阶段深度假设数量
MIN_INTERVAL=2.5                  # 最小深度间隔
DEPTH_INTER_R="4,2,1"             # 深度间隔比例
CR_BASE_CHS="8,8,8"               # 代价体正则化基础通道数
DLOSSW="0.5,1.0,2.0"              # 各阶段深度损失权重

# 学习率调度
LREPOCHS="10,12,14:2"             # 学习率衰减策略

# 日志和保存频率
SUMMARY_FREQ=50                   # 日志记录频率
SAVE_FREQ=1                       # 模型保存频率

# 恢复训练 (可选)
RESUME=false                      # 是否从断点恢复
LOAD_CKPT=""                      # 指定加载的检查点路径 (留空则自动查找最新)

# ==================== 构建命令 ====================
CMD="python train_TLC.py \
    --model ${MODEL} \
    --geo_model ${GEO_MODEL} \
    --gpu_id ${GPU_ID} \
    --dataset_root ${DATASET_ROOT} \
    --logdir ${LOG_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --seed ${SEED} \
    --view_num ${VIEW_NUM} \
    --ref_view ${REF_VIEW} \
    --ndepths ${NDEPTHS} \
    --min_interval ${MIN_INTERVAL} \
    --depth_inter_r ${DEPTH_INTER_R} \
    --cr_base_chs ${CR_BASE_CHS} \
    --dlossw ${DLOSSW} \
    --lrepochs ${LREPOCHS} \
    --summary_freq ${SUMMARY_FREQ} \
    --save_freq ${SAVE_FREQ}"

# 添加 tqdm 参数
if [ "$USE_TQDM" = true ]; then
    CMD="${CMD} --use_tqdm"
    echo "========== 使用 tqdm 进度条模式 =========="
else
    echo "========== 使用纯日志输出模式 =========="
fi

# 添加恢复训练参数
if [ "$RESUME" = true ]; then
    CMD="${CMD} --resume"
    echo "从断点恢复训练"
fi

# 添加检查点加载参数
if [ -n "$LOAD_CKPT" ]; then
    CMD="${CMD} --loadckpt ${LOAD_CKPT}"
    echo "加载检查点: ${LOAD_CKPT}"
fi

# ==================== 执行训练 ====================
echo "============================================"
echo "开始训练: ${MODEL} 模型, ${GEO_MODEL} 几何模型"
echo "GPU: ${GPU_ID}, Batch Size: ${BATCH_SIZE}, Epochs: ${EPOCHS}"
echo "============================================"
echo ""

eval ${CMD}
