"""
srun --immediate=3600    --partition batch,V100-32GB,RTXA6000,RTX3090,RTXB6000,RTXA6000-EI    --gpus=1    --cpus-per-task=4    --mem=80GB    --time=3-00:00:00    --container-workdir="$(pwd)"    --container-image=/netscratch/$USER/pose_image.sqsh    --container-mounts=/netscratch:/netscratch,"$(pwd)":"$(pwd)"    python train.py 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log
srun: jobinfo: version v1.0.0
"""

import os, json, argparse
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler
from config import CFG
from dog_pose_dataset import make_loader, make_target_heatmaps, load_bone_order
from model import DogPoseNet, soft_argmax, MaskedHeatmapLoss, pck, _HAS_TIMM


# Create visualization directories
VIZ_KEYPOINTS_DIR = Path("train_visualizations/keypoints")
VIZ_HEATMAPS_DIR = Path("train_visualizations/heatmaps")
VIZ_KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)




def visualize_all_heatmaps(pred_hm, target_hm, epoch, step, sample_idx):
    N = pred_hm.shape[0]  # 49 joints
    grid_size = int(np.ceil(np.sqrt(N)))  # 7x7 grid for 49
    

    pred_np = pred_hm.detach().cpu().numpy()
    target_np = target_hm.detach().cpu().numpy()
    
    def normalize_hm(hm):
        hm_min = hm.min()
        hm_max = hm.max()
        if hm_max > hm_min:
            return ((hm - hm_min) / (hm_max - hm_min) * 255).astype(np.uint8)
        return np.zeros_like(hm, dtype=np.uint8)

    target_grid = np.zeros((grid_size * 64, grid_size * 48, 3), dtype=np.uint8)
    for idx in range(N):
        row = idx // grid_size
        col = idx % grid_size
        hm_norm = normalize_hm(target_np[idx])
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        target_grid[row*64:(row+1)*64, col*48:(col+1)*48] = hm_color
    

    pred_grid = np.zeros((grid_size * 64, grid_size * 48, 3), dtype=np.uint8)
    for idx in range(N):
        row = idx // grid_size
        col = idx % grid_size
        hm_norm = normalize_hm(pred_np[idx])
        hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        pred_grid[row*64:(row+1)*64, col*48:(col+1)*48] = hm_color
    
    combined = np.hstack([target_grid, pred_grid])
    

    cv2.putText(combined, f"TARGET (ep{epoch} step{step})", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "PREDICTED", (grid_size * 48 + 20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out_path = VIZ_HEATMAPS_DIR / f"ep{epoch:02d}_step{step:06d}_sample{sample_idx}_all_heatmaps.png"
    cv2.imwrite(str(out_path), combined)


def visualize_keypoints(img_batch, pred_coords, gt_coords, visibility, epoch, step, batch_idx):
    """
    Save predicted vs GT keypoints on image (in input space, not original image space).
    """
    BS = img_batch.shape[0]
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
    
    for i in range(min(2, BS)):
        img_np = img_batch[i].cpu().numpy().transpose(1, 2, 0)  # [256, 192, 3]
        img_np = (img_np * IMAGENET_STD + IMAGENET_MEAN)
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        H, W = img_np.shape[:2]  # 256, 192
        
        for j in range(gt_coords.shape[1]):
            if visibility[i, j] > 0:
                x = int(gt_coords[i, j, 0].item() * W)
                y = int(gt_coords[i, j, 1].item() * H)
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(img_np, (x, y), 4, (0, 255, 0), -1)
        
        for j in range(pred_coords.shape[1]):
            x = int(pred_coords[i, j, 0].item() * W)
            y = int(pred_coords[i, j, 1].item() * H)
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(img_np, (x, y), 3, (255, 0, 0), -1)
        
        out_path = VIZ_KEYPOINTS_DIR / f"ep{epoch:02d}_step{step:06d}_sample{i}.png"
        cv2.imwrite(str(out_path), img_np)

def parse_overrides():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--bones")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--bs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--out")
    ap.add_argument("--steps", type=int, default=0, help="run only N train steps (sanity)")
    a = ap.parse_args()
    if a.root:    CFG.paths.root = a.root
    if a.bones:   CFG.paths.bone_order = a.bones
    if a.out:     CFG.paths.out_dir = a.out
    if a.epochs:  CFG.train.epochs = a.epochs
    if a.bs:      CFG.train.batch_size = a.bs
    if a.lr:      CFG.train.lr = a.lr
    if a.workers is not None: CFG.train.num_workers = a.workers
    CFG._sanity_steps = a.steps
    return CFG


def run_epoch(net, loader, crit, device, optim=None, scaler=None, max_steps=None, epoch_num=None):
    train_mode = optim is not None
    net.train(train_mode)
    tot_loss = tot_pck = n = 0
    epoch_start = time.time()
    
    total_steps = len(loader) if max_steps is None else max_steps
    split_name = "Train" if train_mode else "Val"
    
    pbar = tqdm(loader, total=total_steps, desc=f"[Epoch {epoch_num}] {split_name}", 
                unit="batch", leave=False, dynamic_ncols=True, ncols=100)
    
    for step, batch in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break
        
        img = batch["image"].to(device, non_blocking=True)
        kp = batch["keypoints"].to(device, non_blocking=True)
        vis = batch["visibility"].to(device, non_blocking=True)
        targets, tw = make_target_heatmaps(kp, vis)
        targets = targets.to(device)
        tw = tw.to(device)

        with torch.autocast(device_type=device.split(":")[0], enabled=(scaler is not None)):
            hm = net(img)
            loss = crit(hm, targets, tw)

        if train_mode:
            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
            else:
                loss.backward(); optim.step()

        with torch.no_grad():
            pred_coords = soft_argmax(hm)
            batch_pck = pck(pred_coords, kp, vis, thr=CFG.train.pck_thr)
            
            if train_mode and step % 300 == 0:
                visualize_keypoints(img, pred_coords, kp, vis, epoch_num, step, step // 300)
                visualize_all_heatmaps(hm[0], targets[0], epoch_num, step, 0)
        
        bs = img.size(0)
        tot_loss += loss.item() * bs
        tot_pck += batch_pck * bs
        n += bs
        
        if step % 100 == 0:
            pbar.set_postfix({"loss": f"{tot_loss/n:.4f}", "pck": f"{tot_pck/n:.3f}"})
    
    pbar.close()
    epoch_time = time.time() - epoch_start
    return tot_loss / max(n, 1), tot_pck / max(n, 1), epoch_time


def main():
    cfg = parse_overrides()
    os.makedirs(cfg.paths.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bones, _ = load_bone_order(cfg.paths.bone_order)
    cfg.model.num_joints = len(bones)
    print(f"device={device}  timm={_HAS_TIMM}  | {cfg.summary()}")
    print(f"Saving visualizations to:")
    print(f"  Keypoints: {VIZ_KEYPOINTS_DIR}")
    print(f"  Heatmaps: {VIZ_HEATMAPS_DIR}")

    train_loader, _ = make_loader("train", cfg)
    val_loader, _ = make_loader("val", cfg)

    import os as _os
    if cfg.model.backbone == "vit" and cfg.model.vitpose_ckpt and not _os.path.exists(cfg.model.vitpose_ckpt):
        print(f"WARNING: ViTPose ckpt not found at {cfg.model.vitpose_ckpt}")
    net = DogPoseNet(cfg=cfg).to(device)
    crit = MaskedHeatmapLoss()
    optim = AdamW(net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    warm = max(cfg.train.warmup_epochs, 1)
    sched = SequentialLR(
        optim,
        schedulers=[LinearLR(optim, 0.01, 1.0, total_iters=warm),
                    CosineAnnealingLR(optim, T_max=max(cfg.train.epochs - warm, 1))],
        milestones=[warm])
    scaler = GradScaler() if (device == "cuda" and cfg.train.amp) else None
    
    with open(os.path.join(cfg.paths.out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)

    best = -1.0
    for ep in range(1, cfg.train.epochs + 1):
        tr_loss, tr_pck, tr_time = run_epoch(net, train_loader, crit, device, optim, scaler, epoch_num=ep)
        va_loss, va_pck, va_time = run_epoch(net, val_loader, crit, device, epoch_num=ep)
        sched.step()
        print(f"[Epoch {ep:2d}/{cfg.train.epochs}] train {tr_loss:.4f} pck {tr_pck:.3f} ({tr_time/60:5.1f}min) | "
              f"val {va_loss:.4f} pck {va_pck:.3f} ({va_time/60:5.1f}min) | lr {sched.get_last_lr()[0]:.2e}")
        ckpt = {"model": net.state_dict(), "bones": bones, "epoch": ep,
                "val_pck": va_pck, "config": cfg.to_dict()}
        torch.save(ckpt, os.path.join(cfg.paths.out_dir, "last.pt"))
        if va_pck > best:
            best = va_pck
            torch.save(ckpt, os.path.join(cfg.paths.out_dir, "best.pt"))

if __name__ == "__main__":
    main()