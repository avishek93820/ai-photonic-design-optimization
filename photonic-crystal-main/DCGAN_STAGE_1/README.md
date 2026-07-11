#  Stage 1 DCGAN — Photonic Crystal 

Independent PyTorch implementation of the Stage 1 DCGAN architecture described in:
> *A deep learning approach for high-resolution and enhanced efficiency in photonic power dividers*
> Cagatay N. Sengor, Feridun Ay, Cahit Perkgoz — Journal of Applied Physics, March 2025

---

## Overview

This stage implements the Stage 1 DCGAN from the paper above — generating low-resolution (20×20) photonic crystal patterns from 22 real images. Architecture, training pipeline, and dataset loading were implemented independently in PyTorch from the paper's specifications, without reference to the original TensorFlow codebase.

---

## Dataset

- 22 grayscale photonic crystal images (20×20 pixels)
- Silicon regions = 1 (black), Air regions = 0 (white)
- No augmentation applied — trained on raw images only

---

## Architecture

### Generator
| Layer | Details |
|---|---|
| Linear | 300 -> 6400, He init |
| BatchNorm1d + LeakyReLU(0.3) | — |
| Reshape | (batch, 256, 5, 5) |
| ConvTranspose2d | 256->128, k=5, s=1, p=2, BN, LeakyReLU(0.3), He init |
| ConvTranspose2d | 128->64, k=5, s=2, p=2, BN, LeakyReLU(0.3), He init |
| ConvTranspose2d | 64->1, k=5, s=2, p=2, Tanh, Glorot init |
| **Output** | **(batch, 1, 20, 20)** |

### Discriminator
| Layer | Details |
|---|---|
| Conv2d | 1->64, k=5, s=2, p=2, LeakyReLU(0.3), Dropout(0.3), He init |
| Conv2d | 64->128, k=5, s=2, p=2, LeakyReLU(0.3), Dropout(0.3), He init |
| Flatten | 128x5x5 = 3200 |
| Linear | 3200->1, Glorot init |
| **Output** | **(batch, 1) — no Sigmoid** |

No BatchNorm in Discriminator. BatchNorm only in Generator.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Noise dimension | 300 |
| Image size | 20x20 |
| Channels | 1 (grayscale) |
| Batch size | 22 |
| Epochs | 2500 |
| Learning rate | 1e-4 |
| Optimizer | Adam (both G and D) |
| Loss function | BCEWithLogitsLoss |

---

## Training Results

Loss converged to Nash equilibrium (~0.693 = log(2)) around epoch 500 and remained stable through 2500 epochs — indicating the discriminator was 50/50 confused between real and fake images.

```
EPOCH: 500/2500  | LOSS_D: 0.6291 | LOSS_G: 0.8174
EPOCH: 1000/2500 | LOSS_D: 0.6769 | LOSS_G: 0.6864
EPOCH: 1500/2500 | LOSS_D: 0.6735 | LOSS_G: 0.6903
EPOCH: 2000/2500 | LOSS_D: 0.6825 | LOSS_G: 0.7186
EPOCH: 2490/2500 | LOSS_D: 0.6956 | LOSS_G: 0.7347
```

---

## Generated Outputs

**Continuous (raw generator output):**

![Continuous](assets/generated_photonic_continuous.png)

**Binary (threshold > 0.5):**

![Binary](assets/generated_photonic_binary.png)

The binary output is obtained by applying a threshold of 0.5 to the raw output — values above 0.5 become 1 (silicon), below become 0 (air).

---

## Key Differences from Paper

The paper trained on 2000+ efficiency-filtered images generated through an iterative DCGAN + Meep simulation pipeline. This implementation trains directly on 22 raw images without efficiency filtering, as per project constraints. The output shows some mode collapse due to the limited dataset size, which is expected.

---

## Files

| File | Description |
|---|---|
| `DCGAN_Stage1_Photonic_Crystal.ipynb` | Full training notebook |
| `assets/generated_photonic_continuous.png` | Raw generator output |
| `assets/generated_photonic_binary.png` | Binarized output (threshold=0.5) |

---

## Reference

Sengor, C. N., Ay, F., & Perkgoz, C. (2025). A deep learning approach for high-resolution and enhanced efficiency in photonic power dividers. *Journal of Applied Physics*, 137, 124903. https://doi.org/10.1063/5.0255080

Original TensorFlow/Keras implementation by the paper's authors: https://github.com/cagataysengor/DCGAN-and-Photonic-Power-Dividers

---

-> This stage established that 20x20 resolution and paper-spec constraints were insufficient for the actual goal. See `STABILIZED_DCGAN/` for the stage where paper constraints were dropped entirely in favour of the real 34-image dataset at 128x128.
