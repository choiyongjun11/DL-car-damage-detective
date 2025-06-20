import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.models import resnet50
from ultralytics import YOLO

# ── 모델 파일 경로 ─────────────────────────────────────────────────────────────
YOLO_PATH           = "yolo8_last.pt"
PART_MODEL_PATH     = "resnet50_epoch30.pth"
DAMAGE_MODEL_PATH   = "damage_binary_epoch30.pth"

# ── 클래스 이름 정의 ───────────────────────────────────────────────────────────
PART_NAMES          = [
    "Bonnet", "Front bumper", "Front door", "Rear bumper",
    "Head lights", "Rear lamp", "Trunk lid", "Rear door",
    "Rear fender", "Rear Wheel", "Side mirror"
]
DAMAGE_BINARY_NAMES = ["Scratched", "Breakage"]

# ── DataParallel 체크포인트의 'module.' 프리픽스 제거 ────────────────────────────
def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

# ── ResNet50 부품 분류 전용 모델 ────────────────────────────────────────────────
class ResNetPart(torch.nn.Module):
    def __init__(self, num_parts):
        super().__init__()
        base         = resnet50(weights=None)
        self.backbone  = torch.nn.Sequential(*list(base.children())[:-1])
        self.part_head = torch.nn.Linear(2048, num_parts)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.part_head(f)

# ── ResNet50 손상 이진 분류 전용 모델 (학습 코드와 동일하게 cls_head) ──────────
class DamageResNet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base         = resnet50(weights=None)
        self.backbone = torch.nn.Sequential(*list(base.children())[:-1])
        self.cls_head = torch.nn.Linear(2048, 2)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.cls_head(f)

# ── YOLOv8 객체 검출 모델 로드 ─────────────────────────────────────────────────
yolo_model = YOLO(YOLO_PATH)

# ── 부품 분류 모델 로드 (backbone + part_head만 필터링) ─────────────────────────
raw_part_ckpt = strip_module_prefix(torch.load(PART_MODEL_PATH, map_location="cpu"))
part_state = {
    k: v for k, v in raw_part_ckpt.items()
    if k.startswith("backbone.") or k.startswith("part_head.")
}
part_model = ResNetPart(len(PART_NAMES))
part_model.load_state_dict(part_state, strict=True)
part_model.eval()

# ── 손상 분류 모델 로드 (cls_head 포함 전체) ───────────────────────────────────
raw_dmg_ckpt = strip_module_prefix(torch.load(DAMAGE_MODEL_PATH, map_location="cpu"))
dmg_model    = DamageResNet50()
dmg_model.load_state_dict(raw_dmg_ckpt, strict=True)
dmg_model.eval()

# ── 이미지 전처리 정의 ─────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def run_inference(image_path: str, result_path: str):
    image_path = Path(image_path)
    result_path = Path(result_path)

    # 1) 원본 이미지 로드
    pil_img  = Image.open(image_path).convert("RGB")
    orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 2) 부품 분류
    inp = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        p_logits = part_model(inp)
    part_pred = PART_NAMES[p_logits.argmax(1).item()]

    # 3) 손상 분류
    with torch.no_grad():
        d_logits = dmg_model(inp)
    dmg_pred  = DAMAGE_BINARY_NAMES[d_logits.argmax(1).item()]

    # 4) ResNet 결과 시각화
    res_vis = orig_bgr.copy()
    cv2.putText(res_vis, f"Part: {part_pred}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(res_vis, f"Damage: {dmg_pred}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # 5) YOLOv8 예측 & 시각화
    yolo_res  = yolo_model(str(image_path), device="cpu")[0]
    yolo_plot = yolo_res.plot()
    yolo_bgr  = cv2.cvtColor(yolo_plot, cv2.COLOR_RGB2BGR)

    # 6) 높이에 맞춰 리사이즈
    h = orig_bgr.shape[0]
    def resize_to_height(img):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*scale), h))

    orig_v = resize_to_height(orig_bgr)
    res_v  = resize_to_height(res_vis)
    yolo_v = resize_to_height(yolo_bgr)

    # 7) 2×2 그리드 합성
    row1     = np.hstack([orig_v, res_v])
    row2     = np.hstack([orig_v, yolo_v])
    combined = np.vstack([row1, row2])

    # 8) 결과 저장
    cv2.imwrite(str(result_path), combined)

    # 9) 반환 정보
    return {
        "resnet_part":   part_pred,
        "resnet_damage": dmg_pred,
        "yolo_boxes":    len(yolo_res.boxes),
        "yolo_classes":  [int(c) for c in yolo_res.boxes.cls],
    }
