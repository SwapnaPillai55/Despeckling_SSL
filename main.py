#!/usr/bin/env python3
import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
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
EPOCHS, BATCH_SIZE, LR = 50, 8, 1e-5
PATCH_SIZE, EPS = 64, 1e-6
DIM, DEPTH = 96, 2

TRAIN_DIR = "./data/KLSG_train"
TEST_DIR  = "/home/dinesh/swapna/AE_Den/AE_Den/data/test/KLSG_test_sets/noisy_test_dataset"
#TEST_DIR  = "/home/dinesh/swapna/AE_Den/AE_Den/data/SASSED"
BASE_OUT_DIR = Path("./Github_check") 

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

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
        patches = patches.contiguous().view(patches.size(0), patches.size(1), patches.size(2), patches.size(3), -1)
        return patches.median(dim=-1).values

class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)
    def forward(self, x): return self.ln(x.permute(0,2,3,1)).permute(0,3,1,2)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, 4*dim, 1); self.pw2 = nn.Conv2d(4*dim, dim, 1)
        self.act = nn.GELU(); self.gamma = nn.Parameter(1e-6 * torch.ones(dim))
    def forward(self, x): return x + self.gamma.view(1,-1,1,1) * self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))

class IsoConvNeXtDenoiser(nn.Module):
    def __init__(self, dim=96, depth=2):
        super().__init__()
        self.inp = nn.Conv2d(1, dim, 3, padding=1)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depth)])
        self.out = nn.Conv2d(dim, 1, 3, padding=1)
    def forward(self, z): return self.out(self.blocks(self.inp(z)))

# =========================================================
# COMPLEXITY ANALYZER
# =========================================================
def analyze_complexity(model, res=160):
    params = sum(p.numel() for p in model.parameters())
    total_macs = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            macs = (m.kernel_size[0]**2) * m.in_channels * m.out_channels * res * res / m.groups
            total_macs += macs
    total_flops = 2 * total_macs
    print("="*60)
    print(f"COMPLEXITY REPORT @ {res}x{res}")
    print(f"  - Parameters: {params/1e6:.4f} M")
    print(f"  - GMACs:      {total_macs/1e9:.4f}")
    print(f"  - GFLOPs:     {total_flops/1e9:.4f}")
    print("="*60)

# =========================================================
# DATASET & RUNNER
# =========================================================
def targeted_augment(y):
    if torch.rand(1).item() < 0.5:
        L = torch.randint(1, 5, (1,)).item()
        gamma_dist = torch.distributions.Gamma(torch.tensor([float(L)]), torch.tensor([1.0/L]))
        noise = gamma_dist.sample(y.shape).to(y.device).squeeze(-1)
        y = y * noise
    return y.clamp(EPS, 1.0)

class SpeckleTrainDataset(Dataset):
    def __init__(self, root, patch_size):
        self.files = sorted([x for x in Path(root).iterdir() if x.suffix.lower() in {".png", ".jpg", ".tif", ".bmp"}])
        self.patch_size = patch_size; self.to_tensor = T.ToTensor()
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        y = self.to_tensor(Image.open(self.files[idx]).convert("L"))
        y = targeted_augment(y) 
        _, h, w = y.shape
        y = F.pad(y, (0, max(0, self.patch_size-w), 0, max(0, self.patch_size-h)), value=0.0)
        t = torch.randint(0, y.shape[1]-self.patch_size+1, (1,)).item()
        l = torch.randint(0, y.shape[2]-self.patch_size+1, (1,)).item()
        return y[:, t:t+self.patch_size, l:l+self.patch_size]

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
        # Training Logic
        dl = DataLoader(SpeckleTrainDataset(TRAIN_DIR, PATCH_SIZE), batch_size=BATCH_SIZE, shuffle=True, drop_last=True, worker_init_fn=seed_worker)
        med = MedianPool2d(kernel_size=5).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        print("Starting training...")
        for ep in range(EPOCHS):
            beta = b_w * min(1.0, ep / 30)
            net.train()
            for y in dl:
                y = y.to(DEVICE).clamp(EPS, 1.0); z = torch.log(y)
                with torch.no_grad(): z_smf = torch.log(med(y) + EPS)
                z_hat = z - net(z); x_hat = torch.exp(z_hat).clamp(EPS, 1.0)
                
                r = torch.clamp(y/x_hat, 0.25, 4.0); lr = torch.log(r + EPS)
                l_stat = lr.mean((2,3)).abs().mean() + (lr.var((2,3)) - 0.11751).abs().mean() #11751

                gx = torch.abs(x_hat[:,:,:,1:] - x_hat[:,:,:,:-1]); gy = torch.abs(x_hat[:,:,1:,:] - x_hat[:,:,:-1,:])
                wx = torch.exp(-gx/0.01)[:,:,:-1,:]; wy = torch.exp(-gy/0.01)[:,:,:,:-1]

                l_str = (torch.abs(lr[:,:,:,1:] - lr[:,:,:,:-1])[:,:,:-1,:] * wx).mean() + \
                        (torch.abs(lr[:,:,1:,:] - lr[:,:,:-1,:])[:,:,:,:-1] * wy).mean()

                loss = beta * (z_hat - z_smf).abs().mean() + g_w * l_stat + l_w * l_str
                opt.zero_grad(); loss.backward(); opt.step()
        
        torch.save(net.state_dict(), ckpt_path)

    # Inference logic
    net.eval(); to_p = T.ToPILImage()
    print(f"Saving results to {img_dir}...")
    for fp in Path(TEST_DIR).glob("*.*"):
        if fp.suffix.lower() not in [".png", ".jpg", ".tif", ".bmp"]: continue
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
        ("04_Full_Proposed",         0.3, 0.4, 0.15)
    ]

    for i, (name, b, g, l) in enumerate(experiments, 1):
        run_experiment(i, name, b, g, l)

    mode = "Testing" if ONLY_TEST else "Training + Testing"
    print(f"\n{mode} complete.")
