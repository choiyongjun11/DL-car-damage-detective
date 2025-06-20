# inference.py
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.models import resnet50
from ultralytics import YOLO

# 모델 경로
YOLO_PATH         = "yolo8_best.pt"
RESNET_PART_PATH  = "resnet50_epoch30.pth"
RESNET_DMG_PATH   = "damage_binary_epoch30.pth"

# 클래스 이름 정의
PART_NAMES   = [
    "Bonnet", "Front bumper", "Front door", "Rear bumper",
    "Head lights", "Rear lamp", "Trunk lid", "Rear door",
    "Rear fender", "Rear Wheel", "Side mirror"
]
DAMAGE_TYPES = [
    "Scratched", "Separated", "Breakage", "Crushed"
]

# Single-Head ResNet50 정의
class SingleHeadResNet50(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = resnet50(weights=None)
        self.backbone = torch.nn.Sequential(*list(base.children())[:-1])
        self.head     = torch.nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.backbone(x).flatten(1)
        return self.head(features)

# 모델 로드
# YOLOv8

yolo_model    = YOLO(YOLO_PATH)

# ResNet part-classifier
resnet_part   = SingleHeadResNet50(len(PART_NAMES))
full_ckpt     = torch.load(RESNET_PART_PATH, map_location="cpu")
backbone_ckpt = {k.replace('backbone.', ''): v for k, v in full_ckpt.items() if k.startswith('backbone.')}
resnet_part.backbone.load_state_dict(backbone_ckpt)
part_ckpt     = {k.replace('part_head.', ''): v for k, v in full_ckpt.items() if k.startswith('part_head.')}
resnet_part.head.load_state_dict(part_ckpt)
resnet_part.eval()

# ResNet damage-classifier
resnet_dmg    = SingleHeadResNet50(len(DAMAGE_TYPES))
resnet_dmg.backbone.load_state_dict(backbone_ckpt)
dmg_ckpt      = {k.replace('dmg_head.', ''): v for k, v in full_ckpt.items() if k.startswith('dmg_head.')}
resnet_dmg.head.load_state_dict(dmg_ckpt)
resnet_dmg.eval()

# 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# run_inference 함수

def run_inference(image_path: str, result_path: str):
    image_path = Path(image_path)
    result_path = Path(result_path)

    pil_img = Image.open(image_path).convert("RGB")
    orig_rgb = np.array(pil_img)
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    t = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        p_logits = resnet_part(t)
    resnet_part_pred = PART_NAMES[p_logits.argmax(1).item()]

    with torch.no_grad():
        d_logits = resnet_dmg(t)
    resnet_dmg_pred = DAMAGE_TYPES[d_logits.argmax(1).item()]

    resnet_vis = orig_bgr.copy()
    cv2.putText(resnet_vis, f"Part: {resnet_part_pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(resnet_vis, f"Damage: {resnet_dmg_pred}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    yolo_results = yolo_model(str(image_path), device="cpu")[0]
    yolo_plot    = yolo_results.plot()
    yolo_bgr     = cv2.cvtColor(yolo_plot, cv2.COLOR_RGB2BGR)

    h = orig_bgr.shape[0]
    def resize_to_height(img):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    orig_vis   = resize_to_height(orig_bgr)
    resnet_vis = resize_to_height(resnet_vis)
    yolo_vis   = resize_to_height(yolo_bgr)

    row1     = np.hstack([orig_vis, resnet_vis])
    row2     = np.hstack([orig_vis, yolo_vis])
    combined = np.vstack([row1, row2])

    cv2.imwrite(str(result_path), combined)

    return {
        
        "resnet_part": resnet_part_pred,
        "resnet_damage": resnet_dmg_pred,
        "yolo_boxes": len(yolo_results.boxes),
        "yolo_classes": [int(c) for c in yolo_results.boxes.cls]
    }