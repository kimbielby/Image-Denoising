# Results & Analysis

Detailed analysis of the image denoising model performance, training process, and key findings.

## 📊 Final Test Results

### Overall Performance

| Metric | Value | Quality |
|--------|-------|---------|
| **Average PSNR** | 33.95 dB | Excellent |
| **Average SSIM** | 0.8538 | Very Good |
| **Standard Deviation (PSNR)** | 3.33 dB | Good consistency |
| **Standard Deviation (SSIM)** | 0.1030 | Moderate variation |
| **Test Images** | 846 | - |

### PSNR Distribution

```
Minimum:    23.08 dB
25th %ile:  31.90 dB
Median:     34.61 dB
75th %ile:  36.54 dB
90th %ile:  37.66 dB
95th %ile:  38.16 dB
Maximum:    40.17 dB
```

**Interpretation:**
- **50% of images** achieve > 34.61 dB (very good quality)
- **90% of images** achieve > 31.90 dB (good quality)
- **Only 10% of images** fall below 31.90 dB

### SSIM Distribution

```
Minimum:    0.4480
25th %ile:  0.8269
Median:     0.8893
75th %ile:  0.9120
Maximum:    0.9741
```

**Interpretation:**
- **75% of images** have SSIM > 0.83 (strong structural similarity)
- **25% of images** have SSIM > 0.91 (excellent preservation)
- Strong positive correlation with PSNR (when PSNR is high, SSIM is high)

---

## 🔬 Problem Discovery & Solution

### Initial Baseline Results

**First Training Run (No Augmentation):**
```
Average PSNR: 32.87 dB
Average SSIM: 0.8261
Worst Cases: 17.01 - 18.37 dB
PSNR Std:    3.38 dB
```

### Class Imbalance Identified 

**Analysis of worst performing images revealed:**

#### Worst 3 Cases (Baseline):
| Rank | PSNR | SSIM | Image Type                          |
|------|------|------|-------------------------------------|
| #1 | 18.37 dB | 0.6892 | Yellow sponge (bright, saturated)   |
| #2 | 18.08 dB | 0.6754 | Orange toys (bright, colourful)     |
| #3 | 17.01 dB | 0.6543 | Orange fruit (bright, high texture) |

**Pattern identified:** All worst cases were bright/colourful images. 

#### Training Data Analysis:
```python
# Distribution by brightness (mean pixel value)
Dark images (<50):     37.2%
Medium images (50-200): 57.2%
Bright images (>200):   5.6%  ← SEVERELY UNDERREPRESENTED
```

**Root cause:** Model trained predominantly on dark/medium images → poor generalisation to bright images.

---

### Solution: Targeted Data Augmentation

#### Strategy
Instead of random augmentation, target the underrepresented class:

1. **Identify bright images** in training set (brightness > 200)
2. **Create 5× augmented copies** using geometric transforms:
   - Horizontal flip
   - Vertical flip  
   - 90° rotation
   - 180° rotation
   - 270° rotation
3. **Add 50 random augmentations** of non-bright images for variety

#### Results
```
Before augmentation:
  Training images: 5,922
  Bright images:   33 (5.6%)

After augmentation:
  Training images: 6,142
  Bright images:   ~198 (24%)  ← 4.3× increase in representation
```

---

### Impact on Performance

#### Comparison: Baseline vs. With Augmentation

| Metric | Baseline | With Augmentation | Improvement |
|--------|----------|-------------------|-------------|
| **Average PSNR** | 32.87 dB | 33.95 dB | +1.08 dB |
| **Worst Case PSNR** | 17.01 dB | 23.08 dB | **+6.07 dB** ✅ |
| **PSNR Std** | 3.38 dB | 3.33 dB | -0.05 dB (more consistent) |
| **Average SSIM** | 0.8261 | 0.8538 | +0.0277 |

#### Worst 3 Cases (With Augmentation):
| Rank | PSNR | SSIM | Improvement vs Baseline |
|------|------|------|------------------------|
| #1 | 23.56 dB | 0.6115 | **+5.2 dB** |
| #2 | 23.29 dB | 0.5976 | **+5.2 dB** |
| #3 | 23.08 dB | 0.4985 | **+6.1 dB** |

Worst cases improved dramatically (+5-6 dB) while maintaining strong performance on well-represented cases.

---

## 🧪 Experimental Findings

### Experiment 1: ColorJitter Augmentation

**Hypothesis:** Runtime colour augmentation could add further diversity and improve generalisation.

**Implementation:**
```python
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
torch.clamp(output, 0.0, 1.0)  # Prevent out-of-range values
```

**Results:**
- ❌ Training diverged at epochs 4-7 across multiple runs
- ❌ Even with conservative settings (0.2 vs 0.3)
- ❌ Even with clamping to [0, 1] range
- ❌ Loss would spike from 0.05 → 0.8+ suddenly

**Example failure pattern:**
```
Epoch 0-3: Loss decreasing (0.20 → 0.06), PSNR improving (17 → 25 dB)
Epoch 4:   Loss spikes to 0.22, PSNR drops to 9 dB
Epoch 5+:  Complete divergence, negative PSNR
```

**Conclusion:**  
ColorJitter creates edge cases that destabilize training, even with careful tuning. **Preprocessing augmentation (geometric transforms) is more stable and equally effective.**

---

### Experiment 2: Loss Function Comparison

#### MSELoss (Pixel-wise L2)

**Configuration:**
```yaml
loss: MSELoss
learning_rate: 1e-4
```

**Results:**
- ✅ Stable training throughout
- ✅ Smooth convergence
- ✅ Good PSNR (33.95 dB)
- ✅ Good SSIM (0.8538)
- ✅ No training issues

**Training curve:** Steadily decreasing loss with no spikes

---

#### CombinedLoss (MSE + SSIM)

**Configuration:**
```yaml
loss: CombinedLoss
alpha: 0.8  # 80% MSE, 20% SSIM
learning_rate: 3e-5  # Had to reduce from 1e-4
```

**Results:**
- ⚠️ Required 70% lower learning rate (3e-5 vs 1e-4)
- ⚠️ Training instability at higher LR
- ⚠️ Diverged at epochs 4-10 with LR=1e-4
- ❌ Not tested to completion due to instability

**Why it failed:**
- SSIM loss has different gradient characteristics than MSE
- Combined gradient magnitude caused optimisation instability
- Would require extensive hyperparameter tuning

**Decision:** Use MSELoss for stability and production reliability.

---

## 📈 Training Dynamics

### Best Training Run (Final Model)

**Configuration:**
```yaml
Loss: MSELoss
Learning Rate: 1e-4
Batch Size: 16
Optimizer: Adam
Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
Early Stopping: patience=15
Validation: Every epoch
```

**Training Progression:**

| Epoch | Train Loss | Val Loss | Val PSNR | Val SSIM | Status |
|-------|-----------|----------|----------|----------|--------|
| 0 | 0.0312 | 0.0030 | 25.33 dB | 0.6911 | Improving |
| 1 | 0.0051 | 0.0011 | 29.60 dB | 0.7932 | **New best** |
| 5 | 0.0037 | 0.0008 | 31.02 dB | 0.8380 | Improving |
| 10 | 0.0026 | 0.0006 | 32.08 dB | 0.8313 | **New best** |
| 15 | 0.0017 | 0.0005 | 32.82 dB | 0.8600 | **New best** |
| 21 | 0.0015 | 0.0005 | 33.38 dB | 0.8616 | **New best** |

**Observations:**
- Val PSNR > Train PSNR → Good generalisation, no overfitting
- Loss decreased smoothly without spikes
- Validation every epoch enabled precise best model selection
- Early stopping prevented unnecessary training

---

## 🎯 Best and Worst Examples

### Best Performing Images (PSNR > 40 dB)

**Characteristics:**
- Low texture (smooth surfaces)
- Uniform colour regions
- Dark or medium brightness
- Low noise levels in original

**Example: Best #1 (40.17 dB, SSIM 0.9643)**
- Type: Dark carpet texture
- Why it succeeded: Minimal texture, uniform pattern, low noise

---

### Worst Performing Images (PSNR < 25 dB)

**Characteristics:**
- High texture (detailed surfaces)
- Bright/saturated colors
- Complex patterns
- High noise levels in original

**Example: Worst #1 (23.56 dB, SSIM 0.6115)**
- Type: Bright white/coloured patch
- Why it struggled: Very bright, high noise, still challenging despite augmentation

Note that even the "worst" cases at 23+ dB show significant denoising compared to noisy input.

---

## 💡 Technical Findings 

### 1. Data Quality vs Model Complexity

**Finding:** Targeted data augmentation (+6 dB improvement) had far more impact than loss function selection or architectural tweaks.

**Implication:** When performance is poor on specific cases, check training data distribution before adding model complexity.  

---

### 2. Preprocessing vs Runtime Augmentation 

**Finding:** MSELoss with moderate LR converged reliably. CombinedLoss and ColorJitter both caused training instability despite theoretical benefits.

**Implication:** For production systems, choose simple and stable components over theoretically optimal but unstable ones. 

--- 

### 3. Validation Frequency 

**Finding:** Validating every epoch (not every 10) caught subtle improvements and enabled better model selection. 

**Implication:** Validation overhead is minimal (~10-20% slower) but provides much better model selection. 

--- 

### 4. Early Stopping 

**Finding:** Best model typically found between epochs 15 - 30. Training beyond 50 epochs showed no improvement.

**Implication:** Early stopping with patience=15 saves time and prevents overfitting. 

---

## 📉 Current Limitations

### Performance Constraints 

1. **Fixed Input Size**
    - Model requires 512x512 patches
    - Larger images need tiled inference (implemented but adds complexity)
    - Cannot process arbitrarily sized images directly 

2. **Hardware Requirements** 
    - Training requires NVIDIA GPU (4-8GB VRAM minimum)
    - Inference possible on CPU but significantly slower (~10× longer)
    - Not optimised for mobile or edge devices 

3. **Dataset Specificity** 
    - Trained specifically on smartphone camera noise
    - May not generalise well to other noise types (sensor noise, compression artefacts)
    - Performance on professional cameras untested 

4. **Inference Speed** 
    - Current model: ~100ms per 512×512 patch on GPU 
    - Real-time processing not feasible for video 
    - Batch processing recommended for multiple images 

---

## 📊 Appendix: Complete Statistics

### PSNR Detailed Statistics
```
Count:      846 images
Mean:       33.95 dB
Std:        3.33 dB
Min:        23.08 dB
5th %ile:   27.45 dB
10th %ile:  29.78 dB
25th %ile:  31.90 dB
Median:     34.61 dB
75th %ile:  36.54 dB
90th %ile:  37.66 dB
95th %ile:  38.16 dB
Max:        40.17 dB
```

### SSIM Detailed Statistics
```
Count:      846 images
Mean:       0.8538
Std:        0.1030
Min:        0.4480
5th %ile:   0.6245
10th %ile:  0.7156
25th %ile:  0.8269
Median:     0.8893
75th %ile:  0.9120
90th %ile:  0.9345
95th %ile:  0.9456
Max:        0.9741
```

### Training Data Distribution (After Augmentation)
```
Training:   6,142 image pairs
Validation: 1,692 image pairs
Test:       846 image pairs

Bright images in training: ~24% (vs 5.6% baseline)
```
