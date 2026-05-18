# core/custom_trainer.py

from ultralytics.models.yolo.detect import DetectionTrainer
from core.custom_loss import ColorConstrainedLoss
from configs.default_config import PHYSICS_RULES

class CustomDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        # 取得原生的 YOLO 模型
        model = super().get_model(cfg, weights, verbose)
        model.args = self.args
        
        # 將自定義的損失函數注入模型，並傳入物理規則設定
        model.criterion = ColorConstrainedLoss(model, PHYSICS_RULES)
        
        # 確保 criterion 部署在正確的運算裝置上
        device = next(model.parameters()).device
        model.criterion.to(device)
        
        return model