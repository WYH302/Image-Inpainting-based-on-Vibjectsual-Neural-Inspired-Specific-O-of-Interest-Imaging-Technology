# Visual-neural-inspired image inpainting

Training code for specific object-of-interest image reconstruction.

[Training script](trainModel.py) · [Dataset](https://drive.google.com/file/d/10zQYQHpBUjdcDr2g2Zej95Aq_xSbBZtb/view?usp=drive_link) · [License](LICENSE)

## What is available

The readable entry point is `trainModel.py`. It defines the image-pair dataset, EnhancedFSLKNet reconstruction network, loss functions, training loop and PSNR/SSIM monitoring. The original ZIP is retained as an archive; use the readable source when inspecting the implementation.

This checkout has no released trained checkpoint or standalone inference command. A genuine upload-and-reconstruct demo needs a compatible checkpoint and its preprocessing settings. The dataset link is retained from the original release; its availability and redistribution terms have not been independently validated here.

## Set up the training environment

Use an isolated Python environment. Install an appropriate PyTorch/torchvision pair for the intended CPU or CUDA environment, then install the remaining imported packages:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements.txt` records packages imported by the source; it is not a lockfile from a reproduced training run. Do not infer numerical reproducibility from installation alone.

## Prepare paired images

Set these four variables near the top of `trainModel.py`:

| Variable | Input |
| --- | --- |
| `train_folder` | Training input images |
| `target_folder` | Corresponding training targets |
| `validation_folder` | Validation input images |
| `validation_target_folder` | Corresponding validation targets |

The dataset enumerates inputs and targets using sorted filenames. Check pair correspondence, image dimensions and train/validation separation before training. Existing paths refer to the original workstation and must be replaced.

Review `batch_size`, `epochs`, learning rate and the L1/FFT/MSE/VGG loss weights before starting:

```bash
python trainModel.py
```

The script executes training at module scope; do not import it as an inference library. Its VGG perceptual-loss component may require downloading external pretrained VGG weights. Those are not the project's trained reconstruction weights.

## Outputs and evaluation

The script saves `checkpoint_enhanced_epoch*.pth` periodically and eligible `best_model_enhanced_*.pth` files in the working directory. It prints validation loss, PSNR and SSIM. Preserve the configuration and data split alongside each checkpoint.

Metrics that fail inside the current monitoring blocks are caught without stopping training. Check the metric logs before interpreting a completed run as a successful evaluation.

## Real-world Robustness & Edge Deployment

No lighting, occlusion, sensor-noise or edge-device measurements are included in this checkout. A useful next release should evaluate the same fixed checkpoint on unmodified inputs and separately defined perturbations, retain paired outputs and report failure cases as well as averages. Artificial corruptions are stress tests, not a substitute for captured field data.

For deployment, record device, input resolution, batch size, precision, warm-up, median/p95 latency and peak memory. An ONNX export should include a numerical comparison against the PyTorch output. There is currently no verified ONNX artifact or published container image.

## Citation and reuse

Use the associated author-approved paper for bibliographic details and include the repository commit for software provenance. Code reuse is governed by [LICENSE](LICENSE); dataset and pretrained third-party component terms must be checked separately.
