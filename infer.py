import pandas as pd
from ultralytics import YOLO
import supervision as sv
import os
import glob

# 自動尋找最新訓練好的最佳模型權重
print("正在尋找最新訓練好的最佳模型權重...")
weight_files = glob.glob(os.path.join('runs', 'detect', '*', 'weights', 'best.pt'))
if not weight_files:
    raise FileNotFoundError("找不到任何訓練好的權重檔案 (runs/detect/*/weights/best.pt)")

# 取最新修改的檔案
model_path = max(weight_files, key=os.path.getmtime)
print(f"載入模型權重: {model_path}")

model = YOLO(model_path)

# 準備訓練集資料 (路徑調整為本地相對路徑)
dataset_location = "dataset"
ds = sv.DetectionDataset.from_yolo(
    images_directory_path=os.path.join(dataset_location, "train", "images"),
    annotations_directory_path=os.path.join(dataset_location, "train", "labels"),
    data_yaml_path=os.path.join(dataset_location, "data.yaml")
)

all_stats = []

print(f"\n開始對 {len(ds)} 張訓練集照片進行推論...\n")

for img_name, image, target in ds:
    results = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    if len(detections) == 0:
        all_stats.append({
            'Image': img_name,
            'Status': 'No objects detected'
        })
    else:
        for i in range(len(detections)):
            all_stats.append({
                'Image': img_name,
                'Class': model.names[detections.class_id[i]],
                'Confidence': f"{detections.confidence[i]:.2f}",
                'BBox (xyxy)': detections.xyxy[i].tolist()
            })

# 轉換為 DataFrame 方便閱讀
df_results = pd.DataFrame(all_stats)

# 使用 print 取代 Jupyter Notebook 的 display()，因為我們在終端機執行
print(df_results.head(30)) 

# 儲存為 CSV 以供下載
df_results.to_csv('inference_results.csv', index=False)
print("\n推論完成，結果已儲存至 inference_results.csv")
