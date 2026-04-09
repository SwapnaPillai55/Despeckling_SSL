#!/usr/bin/env python3
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# =========================================================
# CONFIG & REPRODUCIBILITY
# =========================================================
ONLY_TEST = False  # Set to True to skip training and load existing .pth files
SEED = 31
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-5

PATCH_SIZE = 64
EPS = 1e-6

DIM = 96
DEPTH = 2

TRAIN_DIR = "./data/KLSG_train"
TEST_DIR  = "/home/dinesh/swapna/AE_Den/AE_Den/data/test/KLSG_test_sets/noisy_test_dataset"
# TEST_DIR = "/home/dinesh/swapna/AE_Den/AE_Den/data/SASSED"
BASE_OUT_DIR = Path("./Github_check")

# Adaptive variance estimation settings
SIGMA2_PATCH = 32
SIGMA2_STRIDE = 16
SIGMA2_KEEP_RATIO = 0.10
SIGMA2_MAX_IMAGES = 200          # limit images for estimation speed
USE_EMA_SIGMA2 = True            # optional online refinement during training
SIGMA2_EMA_MOMENTUM = 0.99
WARMUP_EPOCHS_FOR_EMA = 5        # start EMA refinement after a few epochs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================================================
# MODEL COMPONENTS
# =========================================================
class MedianPool2d(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.k = kernel_size

    def forward(self, x):
        p = self.k // 2
        x = F.pad(x, (p, p, p, p), mode='reflect')
        patches = x.unfold(2, self.k, 1).unfold(3, self.k, 1)
        patches = patches.contiguous().view(
            patches.size(0), patches.size(1), patches.size(2), patches.size(3), -1
        )
        return patches.median(dim=-1).values


class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, 4 * dim, 1)
        self.pw2 = nn.Conv2d(4 * dim, dim, 1)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        return x + self.gamma.view(1, -1, 1, 1) * self.pw2(
            self.act(self.pw1(self.norm(self.dw(x))))
        )


class IsoConvNeXtDenoiser(nn.Module):
    def __init__(self, dim=96, depth=2):
        super().__init__()
        self.inp = nn.Conv2d(1, dim, 3, padding=1)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depth)])
        self.out = nn.Conv2d(dim, 1, 3, padding=1)

    def forward(self, z):
        return self.out(self.blocks(self.inp(z)))


# =========================================================
# COMPLEXITY ANALYZER
# =========================================================
def analyze_complexity(model, res=160):
    params = sum(p.numel() for p in model.parameters())
    total_macs = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            macs = (m.kernel_size[0] ** 2) * m.in_channels * m.out_channels * res * res / m.groups
            total_macs += macs
    total_flops = 2 * total_macs

    print("=" * 60)
    print(f"COMPLEXITY REPORT @ {res}x{res}")
    print(f"  - Parameters: {params / 1e6:.4f} M")
    print(f"  - GMACs:      {total_macs / 1e9:.4f}")
    print(f"  - GFLOPs:     {total_flops / 1e9:.4f}")
    print("=" * 60)


# =========================================================
# ADAPTIVE TARGET VARIANCE ESTIMATION
# =========================================================
def estimate_sigma2(y, patch=32, stride=16, keep_ratio=0.1):
    """
    Estimate log-domain speckle variance from a single noisy image tensor.

    Args:
        y: Tensor of shape [1, H, W] or [1, 1, H, W], assumed in (0, 1].
        patch: patch size for homogeneity analysis
        stride: stride for overlapping patches
        keep_ratio: fraction of lowest-gradient patches retained

    Returns:
        Robust median patch variance in log domain.
    """
    if y.dim() == 4:
        if y.size(0) != 1:
            raise ValueError("estimate_sigma2 expects a single image, not a batch.")
        y = y.squeeze(0)  # [1, H, W]

    if y.dim() != 3 or y.size(0) != 1:
        raise ValueError("estimate_sigma2 expects shape [1, H, W] or [1, 1, H, W].")

    z = torch.log(y.clamp_min(EPS))  # [1, H, W]

    H, W = z.shape[-2:]
    if H < patch or W < patch:
        pad_h = max(0, patch - H)
        pad_w = max(0, patch - W)
        z = F.pad(z, (0, pad_w, 0, pad_h), mode="reflect")

    # extract patches: input to unfold must be [N, C, H, W]
    patches = F.unfold(z.unsqueeze(0), kernel_size=patch, stride=stride)  # [1, patch*patch, N]
    patches = patches.transpose(1, 2)  # [1, N, patch*patch]
    patches = patches.reshape(-1, 1, patch, patch)  # [N, 1, patch, patch]

    # gradient energy as homogeneity score
    gx = patches[..., :, 1:] - patches[..., :, :-1]
    gy = patches[..., 1:, :] - patches[..., :-1, :]
    score = gx.abs().mean(dim=(1, 2, 3)) + gy.abs().mean(dim=(1, 2, 3))

    # select lowest-gradient patches
    k = max(1, int(len(score) * keep_ratio))
    idx = torch.topk(score, k, largest=False).indices
    selected = patches[idx]

    # compute patch variances in log domain
    var = selected.reshape(k, -1).var(dim=1, unbiased=False)

    return var.median().item()


def estimate_dataset_sigma2(
    root,
    patch=32,
    stride=16,
    keep_ratio=0.1,
    max_images=None
):
    """
    Estimate a dataset-level sigma^2 target from noisy images alone.
    Uses robust median over per-image sigma^2 estimates.
    """
    files = sorted(
        [x for x in Path(root).iterdir() if x.suffix.lower() in {".png", ".jpg", ".tif", ".bmp"}]
    )

    if len(files) == 0:
        raise RuntimeError(f"No valid training images found in {root}")

    if max_images is not None:
        files = files[:max_images]

    to_tensor = T.ToTensor()
    sigma2_vals = []

    print(f"Estimating sigma^2_target from {len(files)} training images...")
    for fp in files:
        y = to_tensor(Image.open(fp).convert("L")).clamp(EPS, 1.0)  # [1, H, W]
        sigma2 = estimate_sigma2(
            y,
            patch=patch,
            stride=stride,
            keep_ratio=keep_ratio
        )
        sigma2_vals.append(sigma2)

    sigma2_vals = np.asarray(sigma2_vals, dtype=np.float64)
    sigma2_target = float(np.median(sigma2_vals))

    print(
        f"Estimated sigma^2_target = {sigma2_target:.6f} "
        f"(median over {len(sigma2_vals)} images)"
    )
    print(
        f"  stats: min={sigma2_vals.min():.6f}, "
        f"max={sigma2_vals.max():.6f}, mean={sigma2_vals.mean():.6f}"
    )
    return sigma2_target


# =========================================================
# DATASET & RUNNER
# =========================================================
def targeted_augment(y):
    if torch.rand(1).item() < 0.5:
        L = torch.randint(1, 5, (1,)).item()
        gamma_dist = torch.distributions.Gamma(
            torch.tensor([float(L)]),
            torch.tensor([1.0 / L])
        )
        noise = gamma_dist.sample(y.shape).to(y.device).squeeze(-1)
        y = y * noise
    return y.clamp(EPS, 1.0)


class SpeckleTrainDataset(Dataset):
    def __init__(self, root, patch_size):
        self.files = sorted(
            [x for x in Path(root).iterdir() if x.suffix.lower() in {".png", ".jpg", ".tif", ".bmp"}]
        )
        self.patch_size = patch_size
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        y = self.to_tensor(Image.open(self.files[idx]).convert("L"))
        y = targeted_augment(y)

        _, h, w = y.shape
        y = F.pad(
            y,
            (0, max(0, self.patch_size - w), 0, max(0, self.patch_size - h)),
            value=0.0
        )

        t = torch.randint(0, y.shape[1] - self.patch_size + 1, (1,)).item()
        l = torch.randint(0, y.shape[2] - self.patch_size + 1, (1,)).item()
        return y[:, t:t + self.patch_size, l:l + self.patch_size]


def run_experiment(exp_idx, name, b_w, g_w, l_w):
    print(f"\n>>> EXPERIMENT {exp_idx}: {name}")
    set_seed(SEED)

    img_dir = BASE_OUT_DIR / f"Proposed_{exp_idx}"
    ckpt_dir = BASE_OUT_DIR / name / "model"
    ckpt_path = ckpt_dir / f"{name}_final.pth"

    img_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    net = IsoConvNeXtDenoiser(DIM, DEPTH).to(DEVICE)

    if ONLY_TEST:
        if ckpt_path.exists():
            print(f"Loading weights from {ckpt_path}...")
            net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        else:
            print(f"Warning: {ckpt_path} not found. Skipping experiment.")
            return

    else:
        # -------------------------------------------------
        # 1) Estimate dataset-level sigma^2_target
        # -------------------------------------------------
        sigma2_target = estimate_dataset_sigma2(
            TRAIN_DIR,
            patch=SIGMA2_PATCH,
            stride=SIGMA2_STRIDE,
            keep_ratio=SIGMA2_KEEP_RATIO,
            max_images=SIGMA2_MAX_IMAGES
        )
        running_sigma2 = sigma2_target

        # -------------------------------------------------
        # 2) Training setup
        # -------------------------------------------------
        dl = DataLoader(
            SpeckleTrainDataset(TRAIN_DIR, PATCH_SIZE),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            worker_init_fn=seed_worker
        )

        med = MedianPool2d(kernel_size=5).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=LR)

        print("Starting training...")
        print(f"Initial sigma^2_target = {sigma2_target:.6f}")

        for ep in range(EPOCHS):
            beta = b_w * min(1.0, ep / 30)
            net.train()

            epoch_var_means = []

            for y in dl:
                y = y.to(DEVICE).clamp(EPS, 1.0)
                z = torch.log(y)

                with torch.no_grad():
                    z_smf = torch.log(med(y) + EPS)

                z_hat = z - net(z)
                x_hat = torch.exp(z_hat).clamp(EPS, 1.0)

                # log-ratio residual
                r = torch.clamp(y / x_hat, 0.25, 4.0)
                lr_res = torch.log(r + EPS)

                # per-sample spatial variance
                residual_var = lr_res.var((2, 3), unbiased=False)  # [B, 1]
                current_batch_sigma2 = residual_var.mean().detach().item()
                epoch_var_means.append(current_batch_sigma2)

                # Optional online refinement after warm-up
                if USE_EMA_SIGMA2 and ep >= WARMUP_EPOCHS_FOR_EMA:
                    running_sigma2 = (
                        SIGMA2_EMA_MOMENTUM * running_sigma2
                        + (1.0 - SIGMA2_EMA_MOMENTUM) * current_batch_sigma2
                    )

                sigma2_for_loss = running_sigma2 if USE_EMA_SIGMA2 else sigma2_target

                l_stat = (
                    lr_res.mean((2, 3)).abs().mean()
                    + (residual_var - sigma2_for_loss).abs().mean()
                )

                gx = torch.abs(x_hat[:, :, :, 1:] - x_hat[:, :, :, :-1])
                gy = torch.abs(x_hat[:, :, 1:, :] - x_hat[:, :, :-1, :])
                wx = torch.exp(-gx / 0.01)[:, :, :-1, :]
                wy = torch.exp(-gy / 0.01)[:, :, :, :-1]

                l_str = (
                    (torch.abs(lr_res[:, :, :, 1:] - lr_res[:, :, :, :-1])[:, :, :-1, :] * wx).mean()
                    + (torch.abs(lr_res[:, :, 1:, :] - lr_res[:, :, :-1, :])[:, :, :, :-1] * wy).mean()
                )

                loss = beta * (z_hat - z_smf).abs().mean() + g_w * l_stat + l_w * l_str

                opt.zero_grad()
                loss.backward()
                opt.step()

            epoch_sigma2 = float(np.mean(epoch_var_means)) if len(epoch_var_means) > 0 else float("nan")
            print(
                f"Epoch [{ep+1:03d}/{EPOCHS:03d}] "
                f"beta={beta:.4f} "
                f"batch_sigma2={epoch_sigma2:.6f} "
                f"target_sigma2={running_sigma2:.6f}"
            )

        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "sigma2_target_init": sigma2_target,
                "sigma2_target_final": running_sigma2 if USE_EMA_SIGMA2 else sigma2_target,
                "config": {
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "patch_size": PATCH_SIZE,
                    "dim": DIM,
                    "depth": DEPTH,
                    "sigma2_patch": SIGMA2_PATCH,
                    "sigma2_stride": SIGMA2_STRIDE,
                    "sigma2_keep_ratio": SIGMA2_KEEP_RATIO,
                    "use_ema_sigma2": USE_EMA_SIGMA2,
                    "sigma2_ema_momentum": SIGMA2_EMA_MOMENTUM,
                },
            },
            ckpt_path
        )
        print(f"Saved checkpoint to {ckpt_path}")

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------
    net.eval()
    to_p = T.ToPILImage()
    print(f"Saving results to {img_dir}...")

    for fp in Path(TEST_DIR).glob("*.*"):
        if fp.suffix.lower() not in [".png", ".jpg", ".tif", ".bmp"]:
            continue

        with torch.no_grad():
            y_test = T.ToTensor()(Image.open(fp).convert("L")).unsqueeze(0).to(DEVICE).clamp(EPS, 1.0)
            x_test = torch.exp(torch.log(y_test) - net(torch.log(y_test))).clamp(0, 1)
            to_p(x_test.squeeze(0).cpu()).save(img_dir / f"{fp.stem}_DN{fp.suffix}")


if __name__ == "__main__":
    analyze_complexity(IsoConvNeXtDenoiser(DIM, DEPTH), res=160)

    experiments = [
        ("01_Proposed_NoStat_NoStr", 0.3, 0.0, 0.0),
        ("02_Proposed_NoStr",        0.3, 0.4, 0.0),
        ("03_Proposed_NoStat",       0.3, 0.0, 0.15),
        ("04_Full_Proposed",         0.3, 0.4, 0.15),
    ]

    for i, (name, b, g, l) in enumerate(experiments, 1):
        run_experiment(i, name, b, g, l)

    mode = "Testing" if ONLY_TEST else "Training + Testing"
    print(f"\n{mode} complete.")
