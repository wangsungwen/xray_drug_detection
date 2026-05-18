# train.py

import os
from roboflow import Roboflow
from core.custom_trainer import CustomDetectionTrainer
from configs.default_config import TRAIN_ARGS

def setup_dataset():
    """從本機專案資料夾中讀取資料集"""
    # 假設您的資料夾名稱為 'dataset'，且與 train.py 放在同一個專案根目錄下
    # 您可以依據實際情況修改 "dataset" 這個資料夾名稱
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")

    # 檢查檔案是否存在，避免路徑錯誤導致後續訓練崩潰
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"❌ 找不到資料集設定檔！請確認您的資料集是否已放置於: {dataset_dir}\n"
                                f"並且包含 data.yaml 檔案。")

    print(f"✅ 成功鎖定本機資料集路徑: {data_yaml_path}")
    return data_yaml_path

def main():
    print("啟動物理約束 YOLO 訓練流程...")
    
    # 1. 確認資料集
    # dataset_loc = setup_dataset() 
    # 由於您已經下載過，我們直接使用 TRAIN_ARGS 裡的 data 路徑
    
    # 2. 實例化客製化訓練器
    trainer = CustomDetectionTrainer(overrides=TRAIN_ARGS)
    
    # 3. 開始訓練
    trainer.train()

    # 4. 訓練後檢查
    custom_train_dir = str(getattr(trainer, 'save_dir', f"runs/detect/{TRAIN_ARGS['name']}"))
    weights_path = os.path.join(custom_train_dir, 'weights/last.pt')
    
    if os.path.exists(custom_train_dir):
        print(f"✅ 訓練完成，產出目錄: {custom_train_dir}")
        if os.path.exists(weights_path):
            print(f"✅ 成功生成權重檔案: {weights_path}")
        else:
            print("❌ 未找到權重檔案，請檢查訓練過程是否有報錯。")

if __name__ == "__main__":
    main()