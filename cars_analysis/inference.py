import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.models import resnet50
from ultralytics import YOLO

# 모델 경로
YOLO_PATH   = "yolo8_best.pt"
RESNET_PATH = "resnet50_epoch30.pth"

# 클래스 이름 정의
PART_NAMES   = ["Bonnet", "Front bumper", "Front door", "Rear bumper",
                "Head lights", "Rear lamp", "Trunk lid", "Rear door",
                "Rear fender", "Rear Wheel", "Side mirror"]
DAMAGE_TYPES = ["Scratched", "Separated", "Breakage", "Crushed"]

# ResNet50 멀티헤드 정의
class MultiHeadResNet50(torch.nn.Module):
    def __init__(self, num_parts, num_damages):
        super().__init__()
        base = resnet50(weights=None)
        self.backbone = torch.nn.Sequential(*list(base.children())[:-1])
        self.part_head = torch.nn.Linear(2048, num_parts)
        self.dmg_head  = torch.nn.Linear(2048, num_damages)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.part_head(f), self.dmg_head(f)

# 모델 로드
yolo_model   = YOLO(YOLO_PATH)
resnet_model = MultiHeadResNet50(len(PART_NAMES), len(DAMAGE_TYPES))
resnet_model.load_state_dict(torch.load(RESNET_PATH, map_location="cpu"))
resnet_model.eval()

# 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def run_inference(image_path, result_path):
    image_path = Path(image_path)
    result_path = Path(result_path)

    # 1) 원본 이미지 열기 (RGB → BGR)
    pil_img = Image.open(image_path).convert("RGB")
    orig_rgb = np.array(pil_img)
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    # 2) ResNet50 예측 & 시각화
    t = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        p_logits, d_logits = resnet_model(t)
    resnet_part   = PART_NAMES[p_logits.argmax().item()]
    resnet_damage = DAMAGE_TYPES[d_logits.argmax().item()]

    resnet_vis = orig_bgr.copy()
    cv2.putText(resnet_vis,
                f"ResNet50: {resnet_part}/{resnet_damage}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 3) YOLOv8 예측 & 시각화
    yolo_results = yolo_model(str(image_path), device="cpu")[0]
    # plot() 은 RGB 배열을 반환하니, BGR로 변환
    yolo_plot = yolo_results.plot()
    yolo_bgr  = cv2.cvtColor(yolo_plot, cv2.COLOR_RGB2BGR)

    # 4) 크기 맞추기
    h = orig_bgr.shape[0]
    # 가로 비율 유지
    def resize_to_height(img):
        scale = h / img.shape[0]
        w_new = int(img.shape[1] * scale)
        return cv2.resize(img, (w_new, h))
    resnet_vis = resize_to_height(resnet_vis)
    yolo_vis   = resize_to_height(yolo_bgr)
    orig_vis   = resize_to_height(orig_bgr)

    # 5) 2×2 그리드 합성
    # 가로 방향 Stack: [원본 | ResNet], [원본 | YOLO]
    row1 = np.hstack([ orig_vis, resnet_vis ])
    row2 = np.hstack([ orig_vis, yolo_vis   ])
    # 세로 방향 Stack
    combined = np.vstack([ row1, row2 ])

    # 6) 저장
    cv2.imwrite(str(result_path), combined)

    # 반환 정보 (필요에 따라 템플릿에 전달)
    return {
        "resnet_part":   resnet_part,
        "resnet_damage": resnet_damage,
        "yolo_boxes":    len(yolo_results.boxes),  # 검출된 박스 수
        "yolo_classes":  [int(c) for c in yolo_results.boxes.cls],
    }
