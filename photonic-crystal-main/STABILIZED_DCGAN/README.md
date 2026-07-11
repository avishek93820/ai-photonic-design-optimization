# Stabilized DCGAN — Photonic Crystal

DCGAN trained on 34 photonic crystal images at 128×128 resolution using PyTorch. This is Stage 3 of the architecture progression — the first model to achieve training stability and learn meaningful photonic crystal topology on the full 34-image dataset.

---

## Why This Exists

`DCGAN_STAGE_1` followed a paper specification (20×20, 22 images) and still showed mode collapse with blurry continuous outputs. The resolution was too low to capture the geometric structure of real photonic crystal images.

This stage moves away from the paper entirely. The goal is to push a properly stabilised DCGAN as far as it can go on the real 34-image dataset at a resolution that is actually meaningful — 128×128.

---

## What Changed from DCGAN Stage 1

| Change | Reason |
|---|---|
| Resolution: 20×20 -> 128×128 | 20×20 cannot represent photonic crystal geometry meaningfully |
| Dataset: 22 images -> 34 images | Full dataset used |
| Added `spectral_norm` to all Discriminator Conv2d layers | Constrains D's Lipschitz constant, prevents D from dominating |
| Added `MinibatchStdDev` layer to Discriminator | Penalises mode collapse by making D aware of batch diversity |
| DCGAN weight initialisation (Normal, mean=0, std=0.02) | Stabilises early training dynamics |
| Label smoothing (real labels = 0.9 instead of 1.0) | Prevents D from becoming overconfident on real images |
| BCEWithLogitsLoss | Numerically stable, combines sigmoid and BCE in one operation |

---

## Architecture

### Generator
Input: noise vector (NOISE_DIM=100)

```
ConvTranspose2d: 100 -> 512, 4x4, stride 1  | BatchNorm2d | ReLU
ConvTranspose2d: 512 -> 256, 4x4, stride 2  | BatchNorm2d | ReLU
ConvTranspose2d: 256 -> 128, 4x4, stride 2  | BatchNorm2d | ReLU
ConvTranspose2d: 128 ->  64, 4x4, stride 2  | BatchNorm2d | ReLU
ConvTranspose2d:  64 ->   1, 4x4, stride 2  | Tanh
Output: (1, 128, 128)
```

### Discriminator
Input: (1, 128, 128)

```
spectral_norm(Conv2d):  1 ->  32, 4x4, stride 2 | LeakyReLU(0.2)
spectral_norm(Conv2d): 32 ->  64, 4x4, stride 2 | LeakyReLU(0.2)
spectral_norm(Conv2d): 64 -> 128, 4x4, stride 2 | LeakyReLU(0.2)
spectral_norm(Conv2d):128 -> 256, 4x4, stride 2 | LeakyReLU(0.2)
MinibatchStdDev()
spectral_norm(Conv2d):257 ->   1, 4x4, stride 1
Output: scalar
```

---

## Training Config

| Hyperparameter | Value |
|---|---|
| Image size | 128×128 |
| Channels | 1 (grayscale) |
| NOISE_DIM | 100 |
| BATCH_SIZE | 8 |
| EPOCHS | 300 |
| LR | 2e-4 |
| FEATURE_D / FEATURE_G | 32 |
| Optimizer | Adam, betas=(0.5, 0.999) |
| Loss | BCEWithLogitsLoss |
| Label smoothing | Real labels = 0.9 |

---

## Results

Training stabilised successfully. Discriminator and Generator losses converged to Nash equilibrium (~0.693) and held there through 300 epochs. Zero mode collapse.

The model learned:
- Periodic hole lattice structure
- Waveguide defect channel geometry (diagonal line of missing holes)
- Approximate silicon-air contrast

**Remaining gap:** Outputs are blurry and low contrast compared to real images. Holes blend into the background — edges are not sharp. This is DCGAN's ceiling with 34 images. The distribution-matching problem requires a better loss function.

-> Motivated migration to WGAN-GP.

![Generated Output](assets/output.png)

---

## What This Motivates

BCEWithLogitsLoss measures binary cross-entropy — it tells D to output 0 or 1, which causes training instability when D becomes confident. It also gives G vanishing gradients when D is winning.

The fix is Wasserstein distance — a smoother loss that measures how far apart two distributions are rather than just classifying them. That is WGAN-GP.

-> See `V0_WGAN-GP/` for the next stage.
