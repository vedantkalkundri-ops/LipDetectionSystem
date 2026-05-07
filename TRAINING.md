# 50-Phrase Silent Speech Training (CPU)

## 1) Install dependencies

```bash
python -m pip install torch torchvision torchaudio opencv-python mediapipe numpy onnx onnxruntime
```

## 2) Collect bootstrap videos

```bash
python tools/collect_bootstrap_data.py --speaker-id s01 --clips-per-phrase 3 --language en
python tools/collect_bootstrap_data.py --speaker-id s02 --clips-per-phrase 3 --language en
```

Then assign split values in `data/metadata.csv` (`train`, `val`, `test`).

## 3) Preprocess to lip ROI tensors

```bash
python tools/preprocess_dataset.py --metadata data/metadata.csv --out-dir data/processed
```

## 4) Train baseline model (CPU)

```bash
python tools/train_baseline.py --processed-dir data/processed --epochs 20 --batch-size 6
```

## 5) Evaluate and inspect confusions

```bash
python tools/evaluate_model.py --checkpoint models/lip_phrase_baseline.pt --processed-dir data/processed
```

## 6) Export ONNX for backend

```bash
python tools/export_onnx.py --checkpoint models/lip_phrase_baseline.pt --out models/lip_phrase_baseline.onnx
```

## 7) Run backend with trained model

If `models/lip_phrase_baseline.onnx` exists and `onnxruntime` is installed, `speech_model.py` automatically uses it. Otherwise it falls back to the heuristic recognizer.
