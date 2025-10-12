# Brain Tumor MRI Classification (4 classes)

Custom CNN baseline for classifying brain tumor MRI images into four categories using TensorFlow/Keras. This repo primarily consists of notebooks designed for Google Colab, with reproducible preprocessing, training, and evaluation steps.

Dataset: Kaggle Brain Tumor MRI dataset (Training/Testing split) — classes: `glioma`, `meningioma`, `notumor`, `pituitary`.

Team: Dhanithya Beligolla (SE4050 — Deep Learning Project)

## Project structure

- `PreprocessingData.ipynb` — Data loading, integrity checks, CLAHE enhancement, normalization, stratified splits, augmentation examples, class weights, and saving cleaned arrays.
- `BrainTumor_CustomCNN.ipynb` — End‑to‑end pipeline: dataset download (Colab), robust loader, duplicate removal, stratified split, preprocessing, tf.data pipelines, Custom CNN training, evaluation, artifact saving.


Colab default paths used in the notebooks:
- Project root: `/content/brain_tumor_project` (referred to as `proj_root`)
- Dataset after download: `/content/brain_tumor_project/data/{Training,Testing}`
- Cleaned arrays output: `/content/brain_tumor_project/clean_np`
- Models & metrics: `/content/brain_tumor_project/models`

## End‑to‑end workflow

1) Environment & dataset (Colab)
- GPU info is printed; required Python packages are installed within notebook cells.
- Kaggle API: upload `kaggle.json` when prompted; the notebook downloads and unzips the dataset under `proj_root/data`.

2) Data loading & integrity checks
- Images are read class‑wise from `Training` and `Testing`, converted to RGB, and resized to `224×224`.
- Unreadable files are skipped and logged; basic distribution summaries are shown.

3) Duplicate detection & removal (Training set)
- Perceptual hashing (average hash via `imagehash`) identifies exact duplicates.
- Later duplicates are removed via a keep/remove mask to avoid leakage and bias.

4) Splits & preprocessing
- Use Kaggle’s official `Testing` as the test set; split the `Training` set into train/val with stratification (85/15).
- Normalize pixel values to `[0,1]` (float32). Compute class weights from the training distribution.
- Optional: save cleaned arrays and metadata under `clean_np/`:
	- `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`, and `class_weights.json`.

5) Input pipelines & augmentation
- tf.data pipelines with on‑the‑fly augmentation for training: horizontal/vertical flips, brightness and contrast jitter.
- Batch size defaults to `32`; validation/test datasets are not shuffled and only cast labels.

6) Model — Custom CNN (baseline)
- 3 convolutional blocks (Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout).
- Dense head: 512 → 256 with BatchNorm and Dropout; final Dense softmax with 4 units.
- Input: `(224, 224, 3)`; loss: sparse categorical cross‑entropy; optimizer: Adam (lr=1e‑3); metric: accuracy.

7) Training & callbacks
- Epochs: 40 (configurable). Class weights applied to handle imbalance.
- Custom callback `ValPRF1Callback` computes validation Precision/Recall/F1 (macro) each epoch.
- Checkpoints: best by `val_f1` and best by `val_loss`; EarlyStopping (patience=6, restore best); ReduceLROnPlateau (factor=0.5, patience=3).

8) Evaluation & artifacts
- Metrics on the official test set: Accuracy, macro Precision/Recall/F1; full classification report; confusion matrix heatmap.
- Saved outputs under `proj_root/models/`:
	- `customcnn_final.keras` (final model)
	- `customcnn_best_loss.keras` (best by `val_loss`) and optionally `customcnn_best_f1.keras` (best by `val_f1`)
	- `customcnn_test_metrics.json` (accuracy, precision_macro, recall_macro, f1_macro)
	- `customcnn_history.json` (training history)

## Conventions & reproducibility

- Class order is fixed across the pipeline: `['glioma','meningioma','notumor','pituitary']`. Keep this order for labels, class weights, and reports.
- Determinism: `SEED = 42` is set for Python, NumPy, and TensorFlow.
- Mixed precision: Some cells enable `mixed_float16` for GPUs; use consistently or disable across the run to avoid type mismatches.
- Paths: Notebooks assume Colab paths (`/content`). If running locally, adjust `proj_root` and related path variables early in the notebook.

## Running locally (optional)

These notebooks expect packages installed inside the notebook. For a local environment, ensure at minimum: TensorFlow, NumPy, OpenCV (`cv2`), Pillow, imagehash, scikit‑learn, seaborn, matplotlib. If you want to reproduce the Kaggle download locally, configure the Kaggle CLI and set `proj_root` to a valid local folder, mirroring the `data/Training` and `data/Testing` structure.

## Troubleshooting

- Missing `kaggle.json`: the dataset download cell will prompt you to upload it; obtain it from your Kaggle account settings.
- Path errors (`/content/...` not found): set `proj_root` to your environment and update data paths accordingly.
- `imagehash`/PIL errors: ensure images are readable and packages are installed; unreadable images are skipped by the loader, but hashing still needs PIL to open files.
- Mixed precision issues (NaNs or dtype errors): disable `mixed_float16` or ensure last layer outputs `float32` (the notebook sets the final Dense dtype to `float32`).

## Acknowledgements

- Dataset: Kaggle — masoudnickparvar/brain-tumor-mri-dataset
- Built with TensorFlow/Keras, NumPy, OpenCV, scikit‑learn, Pillow, imagehash, seaborn, matplotlib.

