# tools/active_learning.py

import os
import shutil
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import supervision as sv

# --- 路徑設定 (適應本機專案架構) ---
# 取得專案根目錄 (tools 的上一層)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"

# 自動尋找 runs/detect/ 下最新且包含 best.pt 的權重檔 (支援 train_physics_loss2 等後綴)
def get_latest_model_path():
    runs_dir = PROJECT_ROOT / "runs" / "detect"
    if not runs_dir.exists():
        return PROJECT_ROOT / "runs" / "detect" / "train_physics_loss" / "weights" / "best.pt"
    valid_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "weights" / "best.pt").exists()]
    if not valid_dirs:
        return PROJECT_ROOT / "runs" / "detect" / "train_physics_loss" / "weights" / "best.pt"
    latest_dir = max(valid_dirs, key=lambda x: x.stat().st_mtime)
    return latest_dir / "weights" / "best.pt"

MODEL_PATH = get_latest_model_path()

# 分流輸出的目標資料夾
AL_ROOT = PROJECT_ROOT / "active_learning_split"
REVIEW_PATH = AL_ROOT / "needs_review"
EASY_PATH = AL_ROOT / "high_confidence"

def run_inference_and_evaluate():
    """執行推論並彙整數據"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ 找不到模型權重: {MODEL_PATH}\n請確認是否已經完成訓練。")

    print(f"🔄 載入模型: {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))
    all_stats = []
    
    # 針對訓練集與驗證集進行推論
    splits = ['train', 'valid'] # 如果您的測試集叫做 test，請改為 'test'
    
    for split in splits:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            print(f"⚠️ 略過 {split} 集 (找不到資料夾)")
            continue

        print(f"\n▶️ 開始對 {split} 集進行推論分析...")
        ds = sv.DetectionDataset.from_yolo(
            images_directory_path=str(split_dir / "images"),
            annotations_directory_path=str(split_dir / "labels"),
            data_yaml_path=str(DATA_YAML)
        )

        for img_name, image, target in ds:
            # 降低 inference 的信心門檻，以便擷取低信心值的「懷疑」預測 (預設為 0.25)
            # 設定 conf=0.01，可以讓所有的預測值都顯示出來，方便除錯或是進階分析
            results = model(image, conf=0.01, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # 原始圖片的絕對路徑 (用於後續複製)
            img_abs_path = str(split_dir / "images" / img_name)

            if len(detections) == 0:
                all_stats.append({
                    'Split': split,
                    'Image_Path': img_abs_path,
                    'Image_Name': img_name,
                    'Status': 'No objects detected',
                    'Class': None,
                    'Confidence': None
                })
            else:
                for i in range(len(detections)):
                    all_stats.append({
                        'Split': split,
                        'Image_Path': img_abs_path,
                        'Image_Name': img_name,
                        'Status': 'Detected',
                        'Class': model.names[detections.class_id[i]],
                        'Confidence': float(f"{detections.confidence[i]:.2f}")
                    })

    # 匯出 CSV
    df_results = pd.DataFrame(all_stats)
    csv_out = PROJECT_ROOT / 'inference_results_all.csv'
    df_results.to_csv(csv_out, index=False)
    print(f"\n✅ 推論統計已儲存至: {csv_out}")
    return df_results

def split_dataset(df_results):
    """根據推論結果進行困難樣本分流"""
    print("\n🔍 開始進行主動學習分流 (分離困難樣本)...")
    
    # 建立目錄結構
    for p in [REVIEW_PATH, EASY_PATH]:
        (p / 'images').mkdir(parents=True, exist_ok=True)
        (p / 'labels').mkdir(parents=True, exist_ok=True)

    unique_images = df_results['Image_Path'].dropna().unique()
    review_count, easy_count = 0, 0

    for img_path_str in unique_images:
        img_path = Path(img_path_str)
        if not img_path.exists():
            continue
            
        # 推導標籤路徑 (.jpg 換成 .txt，images 換成 labels)
        label_path = Path(str(img_path).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix('.txt')
        
        # 取得該圖片的所有推論紀錄
        img_records = df_results[df_results['Image_Path'] == img_path_str]
        
        # 困難樣本判斷邏輯
        needs_review = False
        if (img_records['Status'] == 'No objects detected').any():
            needs_review = True
        else:
            if (img_records['Confidence'] < 0.5).any():
                needs_review = True

        # 執行複製分流
        target_dir = REVIEW_PATH if needs_review else EASY_PATH
        shutil.copy(img_path, target_dir / 'images' / img_path.name)
        
        if label_path.exists():
            shutil.copy(label_path, target_dir / 'labels' / label_path.name)
            
        if needs_review: review_count += 1
        else: easy_count += 1

    print("\n🎉 分流作業完成！")
    print(f"👉 需人工複查 (困難樣本): {review_count} 張 -> 存放於 {REVIEW_PATH}")
    print(f"👉 模型已掌握 (高分樣本): {easy_count} 張 -> 存放於 {EASY_PATH}")

if __name__ == "__main__":
    df = run_inference_and_evaluate()
    split_dataset(df)