# CIFAR-10 CNN Debugging

Baseline CNN on CIFAR-10 with:
- Training + validation loops
- Early stopping and best-model saving
- Confusion matrix on validation set
- Failure image export with CSV metadata

## Baseline Model (v1)

- Conv blocks: 2
- Channels: 32 → 64
- Kernel size: 3×3
- Activation: ReLU
- Pooling: MaxPool(2)
- Optimizer: Adam (lr=1e-3)
- Loss: CrossEntropy
- Epochs: 20
- No normalization
- No augmentation