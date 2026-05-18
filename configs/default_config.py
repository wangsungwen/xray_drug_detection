# configs/default_config.py

import os

# 專案與資料集路徑
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "data.yaml")

# 訓練超參數設定
TRAIN_ARGS = {
    'model': 'yolov12n.pt',
    'data': DATASET_PATH,
    'epochs': 300,               # 驗證階段先設小一點，正式訓練可改為 100~300
    'imgsz': 640,
    'batch': 16,
    'name': 'train_physics_loss',
    'mosaic': 1.0,
    'mixup': 0.2,
    # 建議在實作色彩損失時，關閉會扭曲色彩的增強功能
    'hsv_h': 0.0, 
    'hsv_s': 0.0,
    'hsv_v': 0.0
}

# 物理約束設定
PHYSICS_RULES = {
    'penalty_weight': 10.0,
    'drug_classes': {
        'cannabis': 0,
        'heroin': 1,
        'ketamine': 2,
        'mdma': 3
    }
}