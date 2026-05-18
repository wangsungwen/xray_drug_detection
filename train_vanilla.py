# train_vanilla.py

import os
from ultralytics import YOLO
from configs.default_config import TRAIN_ARGS

def main():
    print("啟動標準 (Vanilla) YOLO 訓練流程 (關閉物理/色彩約束及主動學習)...")
    
    # 1. 建立標準的 YOLO 模型 (預設會使用 TRAIN_ARGS 裡指定的 yolov12n.pt 或其他模型)
    model_path = TRAIN_ARGS.get('model', 'yolov8n.pt')
    model = YOLO(model_path)
    
    # 2. 準備訓練參數
    train_kwargs = TRAIN_ARGS.copy()
    
    # 修改輸出資料夾名稱，以便與原先加了物理約束的訓練結果做區分
    train_kwargs['name'] = 'train_vanilla' 
    
    # (選擇性) 如果您原本因為客製化損失函數而關閉了 HSV 增強，現在使用標準訓練，您可以選擇把它開回來
    # 底下這三行會把 HSV 增強參數恢復為 YOLO 預設值，如果您想保持關閉，可將其註解掉
    train_kwargs['hsv_h'] = 0.015
    train_kwargs['hsv_s'] = 0.7
    train_kwargs['hsv_v'] = 0.4
    
    # 3. 開始訓練
    print(f"使用參數: {train_kwargs}")
    results = model.train(**train_kwargs)
    
    # 4. 訓練完成提示
    print("\n✅ 標準 YOLO 訓練完成！")
    print(f"✅ 產出的模型權重 (weights) 及日誌檔 (logs) 已儲存於: {results.save_dir}")

if __name__ == "__main__":
    main()
