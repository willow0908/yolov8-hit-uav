import os, json, shutil, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import optuna
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from ultralytics import YOLO

# ==================== 配置 ====================
ROOT = Path("/root/retest")
DATA_YAML   = ROOT / "dataset.yaml"
IMG_DIR_VAL = ROOT / "images" / "val"
LBL_DIR_VAL = ROOT / "labels" / "val"

MODEL_NAME      = "yolov8s"
BASE_MODEL_PATH = ROOT / "weights" / "yolov8s.pt"
BATCH           = 16              # 与基线一致（基线 batch=16，有效 batch=64）
N_TRIALS        = 100

# ---- 评估口径----
W, H            = 640, 512
NUM_CLASSES     = 4
MATCH_IOU       = 0.5
PRED_CONF       = 0.001
PRED_IOU        = 0.7
MAX_DET         = 300             
N_AP_POINTS     = 101             
CLASS_NAMES     = ["Person", "Car", "Bicycle", "OtherVehicle"]

# ---- 搜索预算 ----
TRIAL_EPOCHS   = 30
TRIAL_FRACTION = 0.25
SEED   = 0
DEVICE = "0"
WORKERS = 8                        

EXP_DIR = ROOT / "runs" / "tune" / MODEL_NAME
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ==================== GT 加载 ====================
lbl_files = sorted(LBL_DIR_VAL.glob("*.txt"))
name2id   = {p.stem: i for i, p in enumerate(lbl_files, start=1)}
gt_by_img = {}
for i, txt in enumerate(lbl_files, start=1):
    rows = []
    for line in open(txt):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        c = int(parts[0])
        if c >= NUM_CLASSES:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        cx, cy, w, h = cx * W, cy * H, w * W, h * H
        rows.append([c, cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    gt_by_img[i] = rows
print(f"GT: {len(gt_by_img)} 张验证图 | 类别数 {NUM_CLASSES} | 目标 mAP@0.5 (101点+包络)")

# ==================== 评估：101 点插值 + precision 包络 ====================
def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
    inter = iw * ih
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0

def ap_from_arrays(det_conf, det_hit, gt_conf, n_point=N_AP_POINTS):
    n_gt = len(gt_conf)
    if n_gt == 0:
        return float("nan")
    gt_sorted  = np.sort(gt_conf)
    det_sorted = np.sort(det_conf)
    hit_sorted = np.sort(det_conf[det_hit]) if det_hit.any() else np.zeros(0)

    ths = np.linspace(0, 1, n_point)
    tp     = n_gt - np.searchsorted(gt_sorted, ths, side="left")
    tp_det = len(hit_sorted) - np.searchsorted(hit_sorted, ths, side="left")
    fp     = (len(det_sorted) - np.searchsorted(det_sorted, ths, side="left")) - tp_det

    denom = tp + fp
    P = np.where(denom > 0, tp / np.maximum(denom, 1e-9), 0.0)
    R = tp / n_gt
    o = np.argsort(R)
    P, R = P[o], R[o]
    for k in range(len(P) - 2, -1, -1):        # precision 包络
        P[k] = max(P[k], P[k + 1])
    return float(np.sum((R[1:] - R[:-1]) * (P[1:] + P[:-1]) / 2))

def ap_per_class(pred_by_img, gt_by_img):
    """
    贪心匹配只在 conf=PRED_CONF 下做一次；
    因匹配按置信度降序，高阈值结果 = 全量匹配的前缀子集，
    故各阈值的 TP/FP 由 searchsorted 解析 —— 比原 21 点版（调 21 次匹配）更快。
    """
    per_det_conf = [[] for _ in range(NUM_CLASSES)]
    per_det_hit  = [[] for _ in range(NUM_CLASSES)]
    per_gt_conf  = [[] for _ in range(NUM_CLASSES)]

    for img_id, gts in gt_by_img.items():
        dets = sorted(pred_by_img.get(img_id, []), key=lambda d: -d[5])
        matched   = [False] * len(gts)
        gt_conf   = [-1.0] * len(gts)
        hit_flags = [False] * len(dets)

        for di, d in enumerate(dets):
            bi, bj = MATCH_IOU, -1
            for j, g in enumerate(gts):
                if matched[j] or g[0] != d[0]:
                    continue
                v = iou_xyxy(d[1:5], g[1:5])
                if v >= bi:
                    bi, bj = v, j
            if bj >= 0:
                matched[bj] = True
                gt_conf[bj] = d[5]
                hit_flags[di] = True

        for di, d in enumerate(dets):
            per_det_conf[d[0]].append(d[5])
            per_det_hit[d[0]].append(hit_flags[di])
        for j, g in enumerate(gts):
            per_gt_conf[g[0]].append(gt_conf[j])

    aps = []
    for c in range(NUM_CLASSES):
        dc = np.asarray(per_det_conf[c], dtype=float)
        dh = np.asarray(per_det_hit[c], dtype=bool)
        gc_ = np.asarray(per_gt_conf[c], dtype=float)
        aps.append(ap_from_arrays(dc, dh, gc_))
    return aps

# ==================== 剪枝 ====================
def pruning_hook(trial):
    def on_fit_epoch_end(trainer):
        m = trainer.metrics.get('metrics/mAP50(B)') if getattr(trainer, 'metrics', None) else None
        if m is None:
            return
        trial.report(float(m), step=trainer.epoch + 1)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return on_fit_epoch_end

# ==================== 13 维搜索空间 ====================
DIMS = [
    ('lr0', 1e-5, 2e-2, True),            #由1e-2改为2e-2
    ('momentum', 0.7, 0.98, False),      # AdamW 下即 β₁；基线 0.9 在范围内
    ('weight_decay', 1e-5, 1e-3, True),
    ('mosaic', 0.0, 1.0, False),
    ('degrees', 0.0, 20.0, False),      
    ('scale', 0.0, 0.5, False),          
    ('fliplr', 0.0, 0.5, False),
    ('box', 5.0, 9.0, False),
    ('cls', 0.2, 2.0, True),
    ('dfl', 0.5, 2.0, False),
    ('lrf', 0.001, 0.5, True),           # ← 下界 0.01 → 0.001（基线 lrf=0.01 原卡边界）
    ('warmup_epochs', 0.0, 5.0, False),
    ('cos_lr', [True, False]),
]
DIM_NAMES = [d[0] for d in DIMS]

def suggest_params(trial):
    p = {}
    for d in DIMS:
        if len(d) == 4:
            p[d[0]] = trial.suggest_float(d[0], d[1], d[2], log=d[3])
        else:
            p[d[0]] = trial.suggest_categorical(d[0], d[1])
    return p

# ==================== 目标函数 ====================
def objective(trial):
    params = suggest_params(trial)
    run_dir = EXP_DIR / f"trial_{trial.number}"
    try:
        model = YOLO(BASE_MODEL_PATH)
        model.add_callback('on_fit_epoch_end', pruning_hook(trial))
        model.train(
            data=str(DATA_YAML),
            epochs=TRIAL_EPOCHS,
            fraction=TRIAL_FRACTION,      # 25% 训练子集（验证仍全量）
            optimizer='AdamW',          
            batch=BATCH, imgsz=640, device=DEVICE, seed=SEED, workers=WORKERS,
            patience=100,
            plots=False, save=True, verbose=False,
            close_mosaic=0,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,   # 红外：关闭颜色增强
            project=str(EXP_DIR), name=f"trial_{trial.number}",
            **params,
        )
        best_pt = run_dir / "weights" / "best.pt"
        if not best_pt.exists():
            return 0.0

        bm = YOLO(str(best_pt))
        rs = bm.predict(source=str(IMG_DIR_VAL), conf=PRED_CONF, iou=PRED_IOU,
                        imgsz=640, device=DEVICE, max_det=MAX_DET, verbose=False)
        pred_by_img = defaultdict(list)
        for r in rs:
            stem = Path(r.path).stem
            if stem not in name2id:
                continue
            img_id = name2id[stem]
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                pred_by_img[img_id].append([int(b.cls.item()), x1, y1, x2, y2,
                                            float(b.conf.item())])
        val = float(np.nanmean(ap_per_class(pred_by_img, gt_by_img)))   # ← nanmean
        print(f"[Trial {trial.number:3d}] mAP50={val:.4f} | "
              + ", ".join(f"{k}={params[k]:.4g}" for k in DIM_NAMES))
        trial.set_user_attr("map50", val)
        return val

    except optuna.TrialPruned:
        raise                             # 必须透传，否则被记成 0.0
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"[OOM] trial {trial.number} 显存不足，跳过")
        return 0.0
    except Exception as e:
        print(f"[Error] trial {trial.number}: {e}")
        return 0.0
    finally:
        shutil.rmtree(run_dir / "weights", ignore_errors=True)   # 清理权重省磁盘
        gc.collect(); torch.cuda.empty_cache()

# ==================== 主程序 ====================
if __name__ == "__main__":
    print(f"🚀 {MODEL_NAME} | 目标 mAP@0.5(101点+包络) | {N_TRIALS} trials | "
          f"25%数据×{TRIAL_EPOCHS}ep | AdamW 对齐基线")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=15),   # ← 5 → 15
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15,               # ← 5 → 15
                                           n_warmup_steps=18,                 # ← 15 → 18
                                           interval_steps=1),
        study_name="tune_yolov8s_map50_v2",     
        storage=f"sqlite:///{EXP_DIR}/optuna.db",
        load_if_exists=True,
    )
    study.optimize(
        objective, gc_after_trial=True,
        callbacks=[MaxTrialsCallback(N_TRIALS,
                   states=(TrialState.COMPLETE, TrialState.PRUNED))],
    )

    print("\n" + "=" * 60)
    print(f"✅ {MODEL_NAME} 搜索完成 | Best mAP50 = {study.best_value:.4f}")
    print(f"Best trial: #{study.best_trial.number}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out = {"model": MODEL_NAME, "batch": BATCH,
           "search_map50": float(study.best_value),
           "best_trial": study.best_trial.number,
           **study.best_params}
    json.dump(out, open(ROOT / f"best_{MODEL_NAME}_map50.json", "w"), indent=2)
    print(f"\n✅ 参数已存档 → {ROOT / f'best_{MODEL_NAME}_map50.json'}")