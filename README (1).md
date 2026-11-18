# Smart Waste Detection: YOLO Training Runs

This archive (`runs.zip`) holds the output of training an object-detection model that spots waste items in images and sorts them into six categories. It is a **training output folder** produced by [Ultralytics YOLO](https://docs.ultralytics.com/). It contains trained weights, metrics and diagnostic plots. It does not contain the dataset or the training script.

## Classes

The model detects 6 classes:

| ID | Class |
|----|-------|
| 0 | Plastic |
| 1 | Metal |
| 2 | Glass |
| 3 | Paper |
| 4 | Cardboard |
| 5 | Organic |

## Folder structure

```
runs/
└── detect/runs/smart_waste/
    ├── yolo26_laptop_train/          # latest run (Sep 30)
    │   ├── args.yaml                 # every training setting used
    │   ├── results.csv               # per-epoch metrics
    │   ├── results.png               # loss and metric curves
    │   ├── weights/
    │   │   ├── best.pt               # best checkpoint (use this for inference)
    │   │   └── last.pt               # most recent checkpoint (use this to resume)
    │   ├── BoxF1_curve.png           # F1 vs. confidence threshold
    │   ├── BoxPR_curve.png           # precision-recall curve
    │   ├── BoxP_curve.png            # precision vs. confidence
    │   ├── BoxR_curve.png            # recall vs. confidence
    │   ├── confusion_matrix.png
    │   ├── confusion_matrix_normalized.png
    │   ├── labels.jpg                # class counts and box-size distribution
    │   ├── train_batch*.jpg          # sample augmented training images with labels
    │   └── val_batch*_labels.jpg / val_batch*_pred.jpg
    │                                 # validation images: ground truth vs. predictions
    └── yolo26_laptop_train.zip       # earlier run (Sep 23), kept as a backup
```

The nested `yolo26_laptop_train.zip` has the same layout. It is a snapshot of the earlier, **completed 30-epoch run**.

## How it works

1. **Data.** Training reads `smart_waste.yaml`, a dataset config that lists the image paths and the six class names. That file is **not** in the archive.
2. **Model.** Training starts from pretrained COCO weights (`yolo11n.pt`, the smallest YOLO11 variant) and fine-tunes them on waste images (transfer learning).
3. **Training.** Settings from `args.yaml`:

   | Setting | Value |
   |---------|-------|
   | Epochs | 30 |
   | Image size | 640 × 640 |
   | Batch size | 4 |
   | Device | CPU (laptop) |
   | Optimizer | `auto` (Ultralytics picks one) |
   | Initial learning rate | 0.01, with 3 warmup epochs |
   | Augmentation | mosaic, horizontal flip, HSV color shifts, scale, translate, random erase, RandAugment |
   | Seed | 0, deterministic |

4. **Validation.** After every epoch the model is scored on the validation split. Results are appended to `results.csv`, and the best epoch by fitness is saved as `weights/best.pt`.
5. **Outputs.** At the end, Ultralytics writes the plots and the sample prediction images.

## Results

From the completed 30-epoch run (`yolo26_laptop_train.zip`):

| Metric | Epoch 1 | Epoch 30 |
|--------|---------|----------|
| mAP@50 | 0.079 | **0.185** |
| mAP@50-95 | 0.053 | **0.143** |
| Precision | 0.278 | 0.187 |
| Recall | 0.158 | 0.308 |

mAP was still rising at epoch 30, so the model had **not converged**. Overall accuracy is low. Some patterns in the confusion matrix of the latest run:

- **Plastic** is detected most reliably, and also gets the most false positives.
- **Glass** is essentially never predicted.
- **Paper** and **Cardboard** have very few correct detections.
- **Organic** items are most often missed and counted as background.

Likely causes are class imbalance, a small model trained for few epochs, and a small batch size. More epochs, more or better-balanced data for the weak classes, and a larger model such as `yolo11s` are the obvious next steps.

## About the two runs

- The **earlier run** (Sep 23) has a full 30-epoch `results.csv` and 5.4 MB weights, which is the expected size for `yolo11n`.
- The **latest run** (Sep 30) has 21 MB weights and only 9 epochs in `results.csv` (mAP@50 about 0.118 at epoch 9). The weights are about four times larger than the earlier run's, so a bigger model may have been used even though `args.yaml` lists `yolo11n.pt`. The run appears to have stopped, or been restarted, before reaching 30 epochs.
- Both runs share the folder name `yolo26_laptop_train`, so the later one reuses the earlier one's name (`exist_ok: true`).

## Using the model

Install Ultralytics:

```bash
pip install ultralytics
```

**Run predictions on images:**

```python
from ultralytics import YOLO

model = YOLO("runs/detect/runs/smart_waste/yolo26_laptop_train/weights/best.pt")
results = model.predict("your_image.jpg", conf=0.25, save=True)
```

**Validate on your dataset:**

```bash
yolo detect val model=path/to/best.pt data=smart_waste.yaml
```

**Continue training:**

```bash
yolo detect train model=path/to/last.pt data=smart_waste.yaml epochs=60
```

Or use `resume=True` to continue an interrupted run.

**Export for deployment:**

```bash
yolo export model=path/to/best.pt format=onnx
```

## Notes

- `args.yaml` contains a Windows path (`C:\Users\...\urop\runs\...`) from the machine where training ran. It is harmless.
- To reproduce training, you need the original `smart_waste.yaml` and its image and label folders in YOLO format.
