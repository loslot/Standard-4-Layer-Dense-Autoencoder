# Dense Autoencoder (4 layers) on MNIST - Denoising

A standard 4-layer dense autoencoder that removes 15% gradient noise from handwritten digits. It is trained on only 2,500 or 5,000 MNIST images and built with PyTorch.

## Project requirements

| Item | Setting |
|---|---|
| Dataset | MNIST, 70,000 images (60,000 train + 10,000 test merged) |
| Subset sizes | 2,500 and 5,000 images |
| Seed | 42 |
| Split | 70% train / 20% test (with 15% gradient noise) / 10% validation |
| Epochs | 50 and 100 |
| Learning rate | 0.01 |
| Evaluation | Training vs validation loss, result plot of 10 digits, confusion matrix |

Every combination of size and epochs is trained, so there are 4 experiments.

## Model

```
Input 28x28 -> Flatten (784)
  Dense 1: 784 -> 512  (BatchNorm + ReLU)   encoder
  Dense 2: 512 -> 256  (BatchNorm + ReLU)   bottleneck
  Dense 3: 256 -> 512  (BatchNorm + ReLU)   decoder
  Dense 4: 512 -> 784  (Sigmoid)            output pixels 0..1
Output 28x28 (the denoised digit)
```

About 1.07 million parameters.

- **Training:** the input is a noisy, slightly shifted or rotated digit, and the target is the clean digit. The loss is MSE.
- **Optimizer:** Adam with learning rate 0.01 and cosine decay. Plain SGD at 0.01 did not train this network in my tests.
- **Test-time augmentation (TTA):** the final result averages the output over the image and its 8 one-pixel shifts.
- **Helper classifier:** an autoencoder does not predict digits. A small classifier reads the denoised images, and its predictions are used for the confusion matrix and the "digit accuracy" columns.

## Files

```
.
├── Dense_Autoencoder_v1.ipynb      the whole project (training, evaluation, graphs)
├── README.md                       this file
├── .gitignore                      should exclude data/ and .venv/
├── data/                           MNIST, downloaded automatically on the first run
└── output/                         everything the notebook produces
```

### Notebook sections (`Dense_Autoencoder_v1.ipynb`)

| Section | What it does |
|---|---|
| 1-2 Install and configuration | Imports and all settings (sizes, epochs, learning rate, noise level, `NOISY_VALIDATION`) |
| 3 Load MNIST | Downloads and merges the 70,000 images |
| 4 Subset and split | Stratified 70/20/10 split, seed 42 |
| 5 Gradient noise | Adds the 15% gradient noise to the test images |
| 6 Model | The 4-layer dense autoencoder |
| 7 Training functions | Augmentation, evaluation, test-time augmentation and the training loop that writes the epoch CSV |
| 8 Helper classifier | Reads denoised images for the confusion matrix |
| 9 Experiments | Trains and evaluates the 4 runs |
| 10-14 Results | Summary table, graphs, summary comparison, classification report, best model |

### Output files (`output/`)

For each of the 4 runs (`size` = 2500 or 5000, `epochs` = 50 or 100):

| File | Content |
|---|---|
| `epochs_size{size}_epochs{epochs}.csv` | One row per epoch (columns below) |
| `graph_loss_size{size}_epochs{epochs}.png` | Training vs validation loss curves |
| `graph_confusion_size{size}_epochs{epochs}.png` | Confusion matrix of digits read from the denoised images |
| `graph_samples_size{size}_epochs{epochs}.png` | 10 random test digits: original, noisy and result |
| `result_10_digits_size{size}_epochs{epochs}.png` | One digit per class 0-9: Original / Test / Result |

For the whole project:

| File | Content |
|---|---|
| `results.csv` | Summary table of all 4 runs |
| `graph_summary_comparison.png` | Error and digit accuracy of all runs side by side |
| `dense_autoencoder_size*_ep*.pt` | Weights of the best model |

### Columns of the epoch CSV

| Column | Meaning |
|---|---|
| Epoch | Training round number |
| Train Loss | MSE between the output and the clean image on the training batches (lower is better) |
| Val Loss | The same MSE on the validation images |
| Train MAE / Val MAE | Average absolute pixel error, on training and validation |
| Test MSE | MSE on the noisy test images, logged every epoch (never used to tune the model) |
| Loss Gap (Val-Train) | Val Loss minus Train Loss |
| Epoch Time (s) | Seconds per epoch |

## Results

Test MSE is measured against the clean image. The noisy image itself has an MSE of about 0.0195, so lower is better. These numbers come from one run on one CPU core, and yours may differ slightly.

| Size / epochs | Test MSE | Test MSE + TTA | PSNR (dB) | Digit accuracy, noisy image | Digit accuracy, denoised image | Time |
|---|---|---|---|---|---|---|
| 2500 / 50 | 0.01195 | 0.01142 | 19.4 | 86.4% | 95.4% | 29 s |
| 2500 / 100 | 0.01017 | 0.00976 | 20.1 | 88.4% | 96.2% | 61 s |
| 5000 / 50 | 0.01045 | 0.01006 | 20.0 | 89.0% | 96.6% | 62 s |
| **5000 / 100** | **0.00887** | **0.00857** | **20.7** | 88.5% | **97.1%** | 126 s |

The best run is 5,000 images with 100 epochs. It cuts the error of the noisy image by about 56%.

## How to run

1. Use Python 3.12 and create a virtual environment:
   ```bash
   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install torch torchvision numpy pandas matplotlib scikit-learn ipykernel
   ```
2. Open `Dense_Autoencoder_v1.ipynb` in VS Code or Jupyter and select the `.venv` kernel.
3. Run all cells. MNIST downloads automatically into `data/` on the first run.

## Notes

- `NOISY_VALIDATION = True` also adds the 15% noise to the validation inputs, so the validation loss tracks the test MSE. Set it to `False` to keep the validation inputs clean.
- The digit accuracy comes from the helper classifier, which is my own addition. It measures how readable the denoised digits are, and it is not the autoencoder's own score.
- The test sets are small (500 and 1,000 images), so small differences between runs are not reliable.
