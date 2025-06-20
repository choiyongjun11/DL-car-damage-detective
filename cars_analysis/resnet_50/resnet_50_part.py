import os
import json
import time
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
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
    imgs, parts, dmgs = zip(*batch)
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    padded = []
    for img in imgs:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        padded.append(F.pad(img, (0, pad_w, 0, pad_h)))
    return torch.stack(padded), torch.tensor(parts), torch.tensor(dmgs)

class DamageDataset(Dataset):
    """
    ROOT_DIR 하위 폴더 구조 예시:
      └─ Bonnet/          (LabelMe JSON + JPEG 혼재, damage 서브폴더 없음)
      └─ Front bumper/    (LabelMe JSON + JPEG 혼재, damage 서브폴더 없음)
      └─ Front door/      (LabelMe JSON + JPEG 혼재, damage 서브폴더 없음)
      └─ 나머지 파트들/   (각 파트 폴더 안에 Scratched/, Breakage/, Separated/, Crushed/ 서브폴더)
           └─ Scratched/ (LabelMe JSON + JPEG)
           └─ Breakage/
           └─ Separated/
           └─ Crushed/
      (Front Wheel 폴더는 아예 사용하지 않음)
    """
    def __init__(self, root_dir, PART_NAMES, DAMAGE_TYPES, scale_factor, transform=None):
        self.root_dir      = root_dir
        self.PART_NAMES    = PART_NAMES
        self.DAMAGE_TYPES  = DAMAGE_TYPES
        self.scale_factor  = scale_factor
        self.transform     = transform
        self.records       = []

        print("=== 데이터셋 탐색 시작 ===")
        for part in PART_NAMES:
            part_dir = os.path.join(root_dir, part)
            if not os.path.isdir(part_dir):
                print(f"[경고] 파트 폴더가 존재하지 않음: {part_dir}")
                continue

            print(f"\n[파트 폴더] {part_dir}")

            # 1) LabelMe JSON이 섞여 있는 파트 (하위에 damage 서브폴더가 없을 때)
            #    ⇒ part_dir 안에 *.json들이 있고, imagePath 키로 이미지 파일명을 지정
            files_root = os.listdir(part_dir)
            json_count_root = sum(1 for f in files_root if f.lower().endswith(".json"))
            if json_count_root > 0:
                print(f"  └─ LabelMe JSON 파일 {json_count_root}개 발견 (서브폴더 없음)")
                for fname in files_root:
                    if not fname.lower().endswith(".json"):
                        continue
                    json_path = os.path.join(part_dir, fname)
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # LabelMe 형식인지 확인
                    if "shapes" in data and "imagePath" in data:
                        # 1. 파트 라벨(shape) 찾기
                        part_idx = None
                        part_bbox = None
                        for shape in data["shapes"]:
                            lbl = shape.get("label", "")
                            # LabelMe JSON의 label이 "Rear_lamp" 형태이면 "_"를 공백으로 바꿔 목록 조회
                            lbl_norm = lbl.replace("_", " ").strip()
                            if lbl_norm == part:
                                pts = shape.get("points", [])
                                if not pts:
                                    continue
                                xs = [p[0] for p in pts]
                                ys = [p[1] for p in pts]
                                x_min, y_min = min(xs), min(ys)
                                x_max, y_max = max(xs), max(ys)
                                w = x_max - x_min
                                h = y_max - y_min
                                part_idx = PART_NAMES.index(part)
                                part_bbox = [x_min, y_min, w, h]
                                break

                        if part_idx is None or part_bbox is None:
                            print(f"    [건너뜀] '{part}' shape을 찾을 수 없음: {json_path}")
                            continue

                        # 2. 데미지 라벨(shape) 찾기
                        dmg_idx = None
                        for shape in data["shapes"]:
                            lbl = shape.get("label", "")
                            if lbl in DAMAGE_TYPES:
                                dmg_idx = DAMAGE_TYPES.index(lbl)
                                break
                        if dmg_idx is None:
                            print(f"    [건너뜀] DAMAGE_TYPES에 해당하는 shape을 찾을 수 없음: {json_path}")
                            continue

                        # 3. LabelMe JSON의 imagePath에서 이미지 파일명 읽기
                        img_file = data.get("imagePath", None)
                        if img_file is None:
                            print(f"    [건너뜀] imagePath 키가 없음: {json_path}")
                            continue

                        img_fullpath = os.path.join(part_dir, img_file)
                        if not os.path.isfile(img_fullpath):
                            print(f"    [건너뜀] 이미지 파일이 존재하지 않음: {img_fullpath}")
                            continue

                        self.records.append({
                            "img":  img_fullpath,
                            "part": part_idx,
                            "dmg":  dmg_idx,
                            "bbox": part_bbox  # [x_min, y_min, width, height]
                        })
                    else:
                        # COCO‐like JSON이 아니라면 건너뜀
                        print(f"    [건너뜀] LabelMe 형식이 아님: {json_path}")

            # 2) damage 서브폴더가 있는 파트들 (예: "Head lights/Scratched", "Head lights/Breakage", ...)
            for dmg in DAMAGE_TYPES:
                dmg_subdir = os.path.join(part_dir, dmg)
                if not os.path.isdir(dmg_subdir):
                    continue
                subfiles = os.listdir(dmg_subdir)
                json_count_sub = sum(1 for f in subfiles if f.lower().endswith(".json"))
                print(f"  └─ 서브폴더 '{dmg}' → LabelMe JSON 파일 {json_count_sub}개")
                for fname in subfiles:
                    if not fname.lower().endswith(".json"):
                        continue
                    json_path = os.path.join(dmg_subdir, fname)
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # LabelMe 형식 파싱
                    if "shapes" in data and "imagePath" in data:
                        # (1) 파트 라벨(shape) 찾기
                        part_idx = None
                        part_bbox = None
                        for shape in data["shapes"]:
                            lbl = shape.get("label", "")
                            lbl_norm = lbl.replace("_", " ").strip()
                            if lbl_norm == part:
                                pts = shape.get("points", [])
                                if not pts:
                                    continue
                                xs = [p[0] for p in pts]
                                ys = [p[1] for p in pts]
                                x_min, y_min = min(xs), min(ys)
                                x_max, y_max = max(xs), max(ys)
                                w = x_max - x_min
                                h = y_max - y_min
                                part_idx = PART_NAMES.index(part)
                                part_bbox = [x_min, y_min, w, h]
                                break

                        if part_idx is None or part_bbox is None:
                            print(f"    [건너뜀] '{part}' shape을 찾을 수 없음: {json_path}")
                            continue

                        # (2) 데미지 라벨은 상위 디렉토리 이름(dmg)에 이미 명시되어 있으므로 그대로 사용
                        dmg_idx = DAMAGE_TYPES.index(dmg)

                        # (3) imagePath에서 이미지 파일명 읽기
                        img_file = data.get("imagePath", None)
                        if img_file is None:
                            print(f"    [건너뜀] imagePath 키가 없음: {json_path}")
                            continue

                        img_fullpath = os.path.join(dmg_subdir, img_file)
                        if not os.path.isfile(img_fullpath):
                            print(f"    [건너뜀] 이미지 파일이 존재하지 않음: {img_fullpath}")
                            continue

                        self.records.append({
                            "img":  img_fullpath,
                            "part": part_idx,
                            "dmg":  dmg_idx,
                            "bbox": part_bbox
                        })
                    else:
                        # COCO‐like JSON 처리 (기존 방식 유지)
                        if "images" in data and "annotations" in data:
                            img_file = data["images"].get("file_name")
                            if img_file is None:
                                continue
                            for ann in data["annotations"]:
                                raw_part = ann.get("repair", [None])[0]
                                dmg_lbl  = ann.get("damage")
                                if raw_part is None or dmg_lbl not in DAMAGE_TYPES:
                                    continue
                                p = raw_part.replace("(L)", "").replace("(R)", "").strip()
                                if p != part or dmg_lbl != dmg:
                                    continue
                                bbox = ann.get("bbox", [0,0,0,0])
                                img_fullpath = os.path.join(dmg_subdir, img_file)
                                if not os.path.isfile(img_fullpath):
                                    continue
                                self.records.append({
                                    "img":  img_fullpath,
                                    "part": PART_NAMES.index(part),
                                    "dmg":  DAMAGE_TYPES.index(dmg_lbl),
                                    "bbox": bbox
                                })
                        else:
                            print(f"    [건너뜀] 알 수 없는 JSON 포맷: {json_path}")

        print(f"\n=== 탐색 완료: 총 레코드 {len(self.records)}개 수집 ===\n")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["img"]).convert("RGB")
        orig_w, orig_h = img.size
        new_w = int(orig_w * self.scale_factor)
        new_h = int(orig_h * self.scale_factor)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        x, y, w, h = r["bbox"]
        x1 = int(x * self.scale_factor)
        y1 = int(y * self.scale_factor)
        w1 = max(int(w * self.scale_factor), 1)
        h1 = max(int(h * self.scale_factor), 1)
        x2 = min(x1 + w1, new_w)
        y2 = min(y1 + h1, new_h)
        if x2 <= x1: x2 = x1 + 1
        if y2 <= y1: y2 = y1 + 1

        crop = img.crop((x1, y1, x2, y2))
        if self.transform:
            crop = self.transform(crop)
        return crop, r["part"], r["dmg"]

class MultiHeadResNet50(nn.Module):
    def __init__(self, num_parts, num_dmg):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone  = nn.Sequential(*list(base.children())[:-1])
        self.part_head = nn.Linear(2048, num_parts)
        self.dmg_head  = nn.Linear(2048, num_dmg)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.part_head(f), self.dmg_head(f)

def evaluate(loader, model, device):
    model.eval()
    p_t, p_p, d_t, d_p = [], [], [], []
    with torch.no_grad():
        for imgs, parts, dmgs in loader:
            imgs, parts, dmgs = imgs.to(device), parts.to(device), dmgs.to(device)
            p_logit, d_logit = model(imgs)
            p_p += p_logit.argmax(1).cpu().tolist()
            p_t += parts.cpu().tolist()
            d_p += d_logit.argmax(1).cpu().tolist()
            d_t += dmgs.cpu().tolist()
    return (
        accuracy_score(p_t, p_p),
        f1_score(p_t, p_p, average="macro"),
        accuracy_score(d_t, d_p),
        f1_score(d_t, d_p, average="macro")
    )

# ─────────────────────────────────────────────────────────────────────────────
def main():
    SCALE_FACTOR = 1
    BATCH_SIZE   = 8
    NUM_EPOCHS   = 30
    LR           = 1e-4
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT_DIR = r"D:\CrackVision\labeld"

    PART_NAMES = [
        "Bonnet", "Front bumper", "Front door",
        "Rear bumper", "Head lights", "Rear lamp",
        "Trunk lid", "Rear door", "Rear fender",
        "Rear Wheel", "Side mirror"
    ]
    DAMAGE_TYPES = ["Scratched", "Separated", "Breakage", "Crushed"]

    criterion_part = nn.CrossEntropyLoss()
    criterion_dmg  = nn.CrossEntropyLoss()

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    ds = DamageDataset(
        ROOT_DIR,
        PART_NAMES,
        DAMAGE_TYPES,
        SCALE_FACTOR,
        transform
    )

    if len(ds) == 0:
        print("❗ 데이터가 하나도 수집되지 않았습니다. 경로 및 JSON 구조를 확인하세요.")
        return

    n_train = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4,
        pin_memory=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4,
        pin_memory=True, collate_fn=pad_collate
    )

    model     = MultiHeadResNet50(len(PART_NAMES), len(DAMAGE_TYPES)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        start = time.time()
        total_loss = 0.0
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"[Epoch {epoch}/{NUM_EPOCHS}]",
            unit="batch"
        )
        for _, (imgs, parts, dmgs) in pbar:
            imgs, parts, dmgs = imgs.to(DEVICE), parts.to(DEVICE), dmgs.to(DEVICE)
            p_logit, d_logit = model(imgs)
            loss = criterion_part(p_logit, parts) + criterion_dmg(d_logit, dmgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(batch_loss=loss.item())

        epoch_time = time.time() - start
        eta = int(epoch_time * (NUM_EPOCHS - epoch))
        p_acc, p_f1, d_acc, d_f1 = evaluate(val_loader, model, DEVICE)
        print(
            f"→ {int(epoch_time)}s | AvgLoss:{total_loss/len(train_loader):.4f} "
            f"| PartAcc:{p_acc:.3f} F1:{p_f1:.3f} "
            f"| DmgAcc:{d_acc:.3f} F1:{d_f1:.3f} | ETA {eta}s"
        )

        ckpt_path = os.path.join(ckpt_dir, f"resnet50_epoch{epoch:02d}.pth")
        torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__":
    freeze_support()
    main()
