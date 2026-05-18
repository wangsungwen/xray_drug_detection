# X 光毒品影像特徵辨識系統 (X-Ray Drug Detection via YOLOv12)

基於 YOLOv12 的輕量化邊緣運算物件偵測系統，專為安檢 X 光機影像中的四類特定管制藥物（大麻、海洛因、K他命、搖頭丸）設計。

本專案不僅使用標準的深度學習訓練流程，更引入了**物理信息規則 (Physics-Informed Rules)** 作為自定義損失函數 (Custom Loss) 來對抗極端複雜背景，並內建了**主動學習 (Active Learning) 分流機制**，以達成持續迭代的 MLOps 循環。

---

## 📂 專案目錄結構

```text
xray_drug_detection/
│
├── configs/
│   └── default_config.py      # 超參數、物理規則權重與路徑設定
│
├── core/
│   ├── custom_loss.py         # 實作 X 光色彩學理約束的 Loss Function
│   └── custom_trainer.py      # 覆寫 YOLOv12 原生訓練器
│
├── dataset/                   # 本機資料集 (需手動放置)
│   ├── train/                 # 訓練集 (images/ & labels/)
│   ├── valid/                 # 驗證集 (images/ & labels/)
│   └── data.yaml              # YOLO 資料集設定檔
│
├── tools/                     
│   └── active_learning.py     # 主動學習推論、數據分析與困難樣本分流腳本
│
├── train.py                   # 專案訓練啟動入口
├── requirements.txt           # 系統套件依賴清單
└── README.md                  # 專案說明文件
```

## 🚀 快速環境建置

為了避免套件衝突，強烈建議使用虛擬環境 (Virtual Environment) 來運行本專案。

### 1. 建立虛擬環境

```bash
# 在專案根目錄下開啟終端機執行
python -m venv xray_env
```

### 2. 啟動虛擬環境

- **Windows (命令提示字元)**: `xray_env\Scripts\activate.bat`
- **Windows (PowerShell)**: `.\xray_env\Scripts\Activate.ps1`
- **macOS / Linux**: `source xray_env/bin/activate`

*(成功啟動後，終端機提示字元前會出現 `(xray_env)`)*

### 3. 安裝依賴套件

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📦 資料集準備

請將您的 X 光影像資料集放置於 `dataset/` 目錄下。

### ⚠️ 極重要注意事項：修改 data.yaml

為確保程式能正確讀取本機資料，請打開 `dataset/data.yaml`，確保內部的路徑設定為相對路徑：

```yaml
# 正確的相對路徑設定範例
train: ./train/images
val: ./valid/images

nc: 4
names: ['Cannabis', 'Heroin', 'Ketamine', 'Mdma']
```

## ⚙️ 系統運作 SOP (MLOps 閉環)

本專案的設計理念為「訓練 ➡️ 診斷 ➡️ 優化」的持續迭代循環：

### Step 1: 啟動模型訓練

讀取 `configs/default_config.py` 中的設定，並載入帶有色彩物理約束的自定義 YOLO 模型進行訓練。

```bash
python train.py
```

- 訓練產出將儲存於：`runs/detect/train_physics_loss/`
- 最佳權重檔位置：`.../weights/best.pt`

### Step 2: 執行主動學習與診斷

訓練完成後，執行此工具對所有資料進行推論，並自動找出模型的「弱點」。

```bash
python tools/active_learning.py
```

執行後將產生兩個重要產出物：

- `inference_results_all.csv`：所有圖片的推論數據統計報告。
- `active_learning_split/` 目錄：系統會自動將圖片分流為：
  - 🟢 `high_confidence/`：模型已確實掌握的樣本。
  - 🔴 `needs_review/`：模型漏抓 (No objects detected) 或信心值偏低 (<0.5) 的困難樣本。

### Step 3: 人工複查與資料回流 (迭代)

1. 前往 `active_learning_split/needs_review/` 檢查被判定為困難的樣本。
2. 檢查是否為人工標註錯誤（修正 txt 檔），或確認該情境特徵過弱。
3. 針對這些弱點情境，收集或合成更多同類型的圖片。
4. 將更新後的資料放回 `dataset/train/`，回到 Step 1 重新訓練。

## 🛠️ 開發與修改指南

- **修改訓練參數** (Epochs, Batch Size等)：請直接編輯 `configs/default_config.py`。
- **修改物理/色彩約束邏輯**：請編輯 `core/custom_loss.py` 中的 `__call__` 函數，實作真實的 PyTorch Tensor 運算與懲罰機制。
- **退出虛擬環境**：開發完畢後，在終端機輸入 `deactivate` 即可恢復全域環境。
