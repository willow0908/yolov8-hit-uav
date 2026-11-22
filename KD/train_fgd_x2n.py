import sys
sys.path.insert(0, '/root/retest/YOLOv8-KD')

from ultralytics import YOLO

# ============ FGD 蒸馏：v8x(教师) → v8n(学生) ============
TEACHER_PT   = '/root/retest/final_train/yolov8x_best/weights/best.pt'
MODEL_PT     = '/root/retest/baseline_results/yolov8n_results/yolov8n_finetune/weights/best.pt'  
DATA_YAML    = '/root/retest/dataset.yaml'

# ---- 完全对齐基线参数 ----
EPOCHS    = 100
IMSZ      = 640
BATCH     = 16
WORKERS   = 8
OPTIMIZER = 'AdamW'
LR0       = 0.01
COS_LR    = False
AMP       = False
IOU       = 0.7
# -------------------------
DEVICE    = '0'
PROJECT   = '/root/retest/distill_results'
NAME      = 'fgd_x2n_yolov8kd'
ALPHA     = 0.01
# =========================================================

model = YOLO(MODEL_PT)

model.train(
    data          = DATA_YAML,
    epochs        = EPOCHS,
    imgsz         = IMSZ,
    batch         = BATCH,
    workers       = WORKERS,
    pretrained    = False,    # 权重已在 MODEL_PT 里加载，不再额外加载
    iou           = IOU,
    device        = DEVICE,
    project       = PROJECT,
    name          = NAME,
    exist_ok      = True,
    optimizer     = OPTIMIZER,
    lr0           = LR0,
    cos_lr        = COS_LR,
    amp           = AMP,
    quant         = False,
    # ---- 知识蒸馏 ----
    Mode          = 'KD',
    KD_Method     = 'FGD',
    teacher       = TEACHER_PT,
    model_pt_path = MODEL_PT,   # 用来推断 LAYER key = yolov8n
    alpha         = ALPHA,
)

# 验证
model.val(
    data   = DATA_YAML,
    imgsz  = IMSZ,
    batch  = BATCH,
    iou    = IOU,
    split  = 'val',
    device = DEVICE,
)
