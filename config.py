from dataclasses import dataclass, field, asdict
from pathlib import Path


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
@dataclass
class Paths:
    root: str = "/netscratch/pinto/face_detection/DOG_UNZIPPED"   # dataset root (all breeds)
    bone_order: str = "bone_order.json"             # frozen 49-bone order
    out_dir: str = "checkpoints"                    # where weights are saved
    cache_dir: str = "."                            # where index_*.pkl live

    def index_cache(self, split):
        return str(Path(self.cache_dir) / f"index_{split}.pkl")


# --------------------------------------------------------------------------
# Image / label geometry  (do NOT change unless the data changes)
# --------------------------------------------------------------------------
@dataclass
class Geometry:
    label_w: int = 1920          # coordinate space of the CSV labels
    label_h: int = 1080
    image_w: int = 960           # actual rendered PNG size
    image_h: int = 540
    input_h: int = 256           # ViTPose top-down input (H x W)
    input_w: int = 192
    hm_h: int = 64               # heatmap size = input / 4 (stride 4)
    hm_w: int = 48

    @property
    def aspect(self):            # target w/h for the crop box
        return self.input_w / self.input_h

    @property
    def scale_x(self):           # label -> image
        return self.image_w / self.label_w

    @property
    def scale_y(self):
        return self.image_h / self.label_h


# --------------------------------------------------------------------------
# Augmentation  (train only; all applied at crop time in the dataset)
# --------------------------------------------------------------------------
@dataclass
class Augment:
    bbox_margin: float = 1.25    # expand tight bbox by this factor
    jitter_center: float = 0.10  # +/- fraction of box w/h
    jitter_scale: float = 0.25   # log-uniform scale +/- this
    flip_p: float = 0.5          # horizontal flip probability
    color_jitter: float = 0.30   # brightness/contrast/saturation +/-
    # rotation/scale intentionally OFF: camera angle + focal already cover them


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
@dataclass
class Model:
    backbone: str = "vit"                       # "vit" (timm ViT-B) or "simple"
    vit_name: str = "vit_base_patch16_224"
    pretrained: bool = True                      # ImageNet init if no vitpose_ckpt
    vitpose_ckpt: str = "/netscratch/pinto/face_detection/vitpose_base.pth"  # "" to disable
    vitpose_expert: int = 3                      # ViTPose++ MoE expert: 3=AP10K (animal)
    num_joints: int = 49                         # overridden from bone_order.json
    deconv_hidden: int = 256
    n_deconv: int = 2                            # 16->64 (x4)


# --------------------------------------------------------------------------
# Heatmap target / loss
# --------------------------------------------------------------------------
@dataclass
class Target:
    sigma: float = 2.0         # Gaussian blob std (in heatmap px)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class Train:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 2
    num_workers: int = 4
    amp: bool = True             # mixed precision on CUDA
    pck_thr: float = 0.1         # PCK@thr for eval
    seed: int = 42
    split_ratios: tuple = (0.8, 0.1, 0.1)   # train / val / test by ACTION


# --------------------------------------------------------------------------
# Master config
# --------------------------------------------------------------------------
@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    geom: Geometry = field(default_factory=Geometry)
    aug: Augment = field(default_factory=Augment)
    model: Model = field(default_factory=Model)
    target: Target = field(default_factory=Target)
    train: Train = field(default_factory=Train)

    def to_dict(self):
        return asdict(self)

    def summary(self):
        g = self.geom
        return (f"input {g.input_h}x{g.input_w} -> hm {g.hm_h}x{g.hm_w} | "
                f"joints {self.model.num_joints} | backbone {self.model.backbone} | "
                f"bs {self.train.batch_size} lr {self.train.lr} epochs {self.train.epochs}")


CFG = Config()


if __name__ == "__main__":
    import json
    print(CFG.summary())
    print(json.dumps(CFG.to_dict(), indent=2, default=str))