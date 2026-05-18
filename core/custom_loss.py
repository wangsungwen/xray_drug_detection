# core/custom_loss.py

import torch
import torch.nn as nn
from ultralytics.utils.loss import v8DetectionLoss

class ColorConstrainedLoss(v8DetectionLoss, nn.Module):
    def __init__(self, model, config_rules):
        nn.Module.__init__(self)
        super().__init__(model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 讀取來自設定檔的物理規則參數
        self.penalty_weight = torch.tensor(config_rules['penalty_weight'], device=self.device)
        self.heroin_class_id = config_rules['drug_classes']['heroin']

    def __call__(self, preds, batch):
        # 動態同步裝置 (解決 YOLO v8DetectionLoss 內部變數最初在 CPU 上建立的問題)
        current_device = preds['boxes'].device if isinstance(preds, dict) else preds[0]['boxes'].device if isinstance(preds, tuple) and isinstance(preds[0], dict) else preds[1]['boxes'].device
        if getattr(self, 'device', None) != current_device:
            self.device = current_device
            self.penalty_weight = self.penalty_weight.to(current_device)
            if hasattr(self, 'proj'):
                self.proj = self.proj.to(current_device)
            if hasattr(self, 'bbox_loss'):
                self.bbox_loss = self.bbox_loss.to(current_device)
            if hasattr(self, 'bce'):
                self.bce = self.bce.to(current_device)

        # 取得原生的 YOLO Loss
        loss, loss_items = super().__call__(preds, batch)

        # ---------------------------------------------------------
        # 物理色彩約束邏輯 (Physics-based Color Constraint)
        # ---------------------------------------------------------
        # TODO: 將此處替換為真實的 Tensor 運算邏輯
        # 1. 解碼 preds 取得 Bounding Box 座標
        # 2. 從 batch['img'] 中裁切出物件區域
        # 3. 計算該區域的色彩均值
        # 4. 與學理上的橘/棕色張量計算 MSE 差異
        
        # 目前的佔位測試 (Dummy Penalty)
        heroin_mask = (batch['cls'] == self.heroin_class_id).float()
        dummy_penalty = heroin_mask.mean() * self.penalty_weight 
        
        # 將自定義懲罰加入總 Loss 中
        total_loss = loss + dummy_penalty
        
        # 需確保回傳的 loss_items 維度與原生 YOLO 一致，以免報錯
        # 此處僅為示意，實務上可能需要將 penalty 記錄進 loss_items 的特定欄位以供 Tensorboard 監控
        
        return total_loss, loss_items