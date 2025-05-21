## Instructions for Training and Evaluating DeiT on CIFAR-10

This document provides step-by-step instructions to set up the environment, prepare the CIFAR-10 dataset, train a DeiT model from scratch, fine-tune a pre-trained DeiT model, and evaluate results.

---

### 1. Requirements

**Python packages** (install via `pip`):

```bash
pip install torch torchvision timm torchmetrics matplotlib
```

---

### 2. Directory Structure

```
project-root/
├── instruction.md
├── training.py         # Main training script

```

---

### 3. Environment Variables

Set the GPU device to use (e.g., GPU 4) (already in the code)


### 4. Data Preparation

The script automatically downloads CIFAR-10 and applies transformations:

* **Training set transforms**:

  * Resize to 224x224
  * Random horizontal flip
  * ToTensor
  * Normalize (mean=0.5, std=0.5)

* **Test/Validation set transforms**:

  * Resize to 224x224
  * ToTensor
  * Normalize (mean=0.5, std=0.5)

The dataset is split into:

* 45,000 training samples
* 5,000 validation samples
* 10,000 test samples

---

### 5. Training from Scratch

1. **Hyperparameters**

   * Batch size: 128
   * Image size: 224
   * Number of epochs: 300
   * Learning rate: 3e-4
   * Weight decay: 1e-4

2. **Model**

   * `deit_tiny_patch16_224` created with `pretrained=False`, `num_classes=10`.

3. **Loss & Optimizer**

   * Loss: CrossEntropyLoss
   * Optimizer: AdamW

4. **Run training**

    * kick up the Training.py

5. **Checkpoint**

   * Model saved as `deit_tiny_cifar10_scratch_2.pth`.

---

### 6. Fine-Tuning Pre-trained Model

1. **Hyperparameters**

   * Fine-tune for 5 epochs
   * Learning rate: 1e-4

2. **Model**

   * `DeiT-base distilled` created with `pretrained=True`, `num_classes=10`.

3. **Optimizer**

   * AdamW on all parameters

4. **Run fine-tuning**

    * kick up the Training.py

---

### 7. Evaluation

After training or fine-tuning, the script prints test accuracy:

```bash
# Scratch-trained accuracy
Scratch-trained DeiT-tiny → CIFAR-10 test accuracy: XX.XX%

# Fine-tuned accuracy
Pre-trained → fine-tuned DeiT-tiny test accuracy: YY.YY%
```

---

### 8. Plotting Training Curves

The script plots training and validation accuracy over epochs using Matplotlib. Ensure you have a display backend or save the plot to a file by modifying the script:

```python
plt.savefig('accuracy_curve.png')
```
