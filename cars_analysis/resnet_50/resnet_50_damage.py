import os
import json
import time
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from multiprocessing import freeze_support
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
def pad_collate(batch):
    """
    배치 내 이미지 크기가 서로 다를 때, 가장 큰 높이/너비에 맞춰서 패딩해 줍니다.
    """
    imgs, labels = zip(*batch)
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    padded = []
    for img in imgs:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # (left, right, top, bottom) 순서
        padded.append(F.pad(img, (0, pad_w, 0, pad_h)))
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)


class DamageDataset(Dataset):
    """
    단일 폴더 내부에 섞여 있는 모든 LabelMe‐style JSON과 이미지(.jpg)에서
    → JSON 내부의 "images"와 "annotations" 키를 읽어서 “파손 유형”만 추출합니다.
    → 하나의 JSON 안에 여러 개의 annotation(파손 정보)이 들어 있으므로,
      - 'Scratched' → label 0
      - ['Separated', 'Breakage', 'Crushed'] → label 1
    → 반환: (원본 이미지를 scale_factor만큼 리사이즈 + transform 처리한 Tensor, binary_label)
    """
    def __init__(self, folder, scale_factor, transform=None):
        super().__init__()
        self.folder       = folder
        self.scale_factor = scale_factor
        self.transform    = transform
        self.records      = []

        print("=== DamageDataset 탐색 시작 ===")
        for fname in os.listdir(folder):
            if not fname.lower().endswith(".json"):
                continue

            json_path = os.path.join(folder, fname)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  [SKIP] JSON 파싱 실패: '{fname}' → {e}")
                continue

            # 1) JSON에 "images" 및 "annotations" 키가 모두 존재하는지 확인
            if "images" not in data or "annotations" not in data:
                print(f"  [SKIP] '{fname}': 'images' 또는 'annotations' 키 없음")
                continue

            # 2) 이미지 파일명 얻기
            img_info = data["images"]
            if not isinstance(img_info, dict) or "file_name" not in img_info:
                print(f"  [SKIP] '{fname}': 'images.file_name' 키 없음")
                continue

            img_file = img_info["file_name"]
            img_fullpath = os.path.join(folder, img_file)
            if not os.path.isfile(img_fullpath):
                print(f"  [SKIP] '{fname}': 이미지 파일 '{img_file}'을(를) 찾을 수 없음")
                continue

            # 3) annotation 배열을 순회하며 라벨을 이진화
            ann_list = data["annotations"]
            if not isinstance(ann_list, list):
                print(f"  [SKIP] '{fname}': 'annotations'가 리스트가 아님")
                continue

            added_any = False
            for ann in ann_list:
                # ann 딕셔너리에 "damage" 키가 있는지 확인
                dmg_lbl = ann.get("damage", None)
                if not isinstance(dmg_lbl, str):
                    continue
                dmg_lbl = dmg_lbl.strip()

                if dmg_lbl == "Scratched":
                    bin_idx = 0
                elif dmg_lbl in ("Separated", "Breakage", "Crushed"):
                    bin_idx = 1
                else:
                    continue

                # annotation 단위로 하나의 레코드를 추가
                self.records.append({
                    "img": img_fullpath,
                    "label": bin_idx
                })
                added_any = True

            if not added_any:
                # 이 JSON 파일에 Scratched/Separated/Breakage/Crushed 중 어느 것도 없으면 skip
                print(f"  [SKIP] '{fname}': 대상 damage 레이블 없음")

        print(f"=== 탐색 완료: 총 {len(self.records)}개 레코드 수집 ===\n")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["img"]).convert("RGB")

        # 전체 이미지를 scale_factor만큼 리사이즈
        orig_w, orig_h = img.size
        new_w = int(orig_w * self.scale_factor)
        new_h = int(orig_h * self.scale_factor)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        if self.transform:
            img = self.transform(img)
        return img, r["label"]


class DamageResNet50(nn.Module):
    """ResNet-50 기반으로 이진 분류 (Scratched vs Breakage)"""
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        # 이진 분류: 출력 차원 2
        self.cls_head = nn.Linear(2048, 2)

    def forward(self, x):
        f = self.backbone(x).flatten(1)   # [B, 2048, 1, 1] → [B, 2048]
        return self.cls_head(f)          # [B, 2]


def evaluate(loader, model, device):
    """
    Validation 등에서 전체 Accuracy와, 클래스별 Precision/Recall/F1/Support를 계산하여 반환
    """
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            pred = logits.argmax(1)
            all_preds += pred.cpu().tolist()
            all_targets += labels.cpu().tolist()

    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds,
        labels=[0, 1],
        zero_division=0
    )
    return acc, precision, recall, f1, support


# ─────────────────────────────────────────────────────────────────────────────
def main():
    SCALE_FACTOR = 1
    BATCH_SIZE   = 8   # RTX 4070 8GB 기준. OOM 시에는 4로 줄여 보세요.
    NUM_EPOCHS   = 30
    LR           = 1e-4
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = r"D:\CrackVision\7. by_model_damage_date\아반떼-AD\filtered_AD"

    # 1) 데이터셋 생성
    transform = transforms.Compose([
        transforms.Resize((224, 224)),          # 메모리 절약을 위해 224×224로 고정
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    ds = DamageDataset(
        DATA_DIR,
        scale_factor=SCALE_FACTOR,
        transform=transform
    )

    if len(ds) == 0:
        print("❗ Damage 데이터가 전혀 수집되지 않았습니다. 경로와 JSON 구조를 확인하세요.")
        return

    # 2) train/val split (80:20)
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True,  num_workers=2,
        pin_memory=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2,
        pin_memory=True, collate_fn=pad_collate
    )

    # 3) 모델, 손실 함수, 옵티마이저, 스케줄러
    model = DamageResNet50().to(DEVICE)
    # → 클래스 균형 가중치 없이 단순 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    ckpt_dir = "./checkpoints_damage_binary"
    os.makedirs(ckpt_dir, exist_ok=True)

    # 4) 학습 루프
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        start = time.time()
        total_loss = 0.0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"[Epoch {epoch}/{NUM_EPOCHS}]",
            unit="batch"
        )
        for _, (imgs, labels) in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(batch_loss=loss.item())

        scheduler.step()  # 매 10 에폭마다 lr *= 0.1

        epoch_time = time.time() - start
        eta = int(epoch_time * (NUM_EPOCHS - epoch))

        # 5) Validation → 전체 Accuracy + 클래스별 Precision/Recall/F1/Support 출력
        acc, precision, recall, f1, support = evaluate(val_loader, model, DEVICE)
        print(f"→ {int(epoch_time)}s | AvgLoss:{total_loss/len(train_loader):.4f} | Acc:{acc:.3f} | ETA {eta}s")
        print("   └─ 클래스별 → Precision / Recall / F1 / Support")
        print(f"      Scratched(0) → P:{precision[0]:.3f}  R:{recall[0]:.3f}  F1:{f1[0]:.3f}  Sup:{support[0]}")
        print(f"      Breakage(1)  → P:{precision[1]:.3f}  R:{recall[1]:.3f}  F1:{f1[1]:.3f}  Sup:{support[1]}")

        # 6) 에폭별 체크포인트 저장
        ckpt_path = os.path.join(ckpt_dir, f"damage_binary_epoch{epoch:02d}.pth")
        torch.save(model.state_dict(), ckpt_path)

    print("=== 학습 완료 ===")


if __name__ == "__main__":
    freeze_support()
    main()
