# Per-Epoch Features

这是一个可独立复制的 PSG 特征提取包。外部项目只需复制整个
`per_epoch_features` 文件夹，不需要复制原项目的其他 Python 文件。

## 安装依赖

```bash
pip install -r per_epoch_features/requirements.txt
```

## 从外部 Python 调用

保证 `per_epoch_features` 的父目录位于 `PYTHONPATH`，然后：

```python
from per_epoch_features import extract_features

features = extract_features(
    data_folder=r"D:\physionet2026\training_set",
    bids_folder="sub-I0002150005420",
    site_id="I0002",
    session_id=1,
)

X_seq = features["X_seq"]          # (N_epochs, 483), float32
X_ecg = features["X_ecg"]          # (N_windows, 37), float32
x_static = features["x_static"]    # (196,), float32
mask = features["mask"]            # (N_epochs,), bool
```

如果包不在当前工程目录，可以显式加入父目录：

```python
import sys
sys.path.insert(0, r"D:\path\containing_the_package")

from per_epoch_features import extract_features
```

## 保存和加载

```python
from per_epoch_features import load_features, save_features

save_features(features, "record_features.npz")
features = load_features("record_features.npz")
```

## 命令行调用

在包的父目录运行：

```bash
python -m per_epoch_features \
  --data-folder /path/to/training_set \
  --bids-folder sub-I0002150005420 \
  --site-id I0002 \
  --session-id 1 \
  --output record_features.npz
```

## 数据目录

```text
training_set/
├── demographics.csv
├── physiological_data/
│   └── {SiteID}/
│       └── {BidsFolder}_ses-{SessionID}.edf
└── algorithmic_annotations/
    └── {SiteID}/
        └── {BidsFolder}_ses-{SessionID}_caisr_annotations.edf
```

ECG 缺失时，`X_ecg` 返回 `(0, 37)` 空数组，`mask` 全为 `False`。算法标注
或构造时序特征所需的通道缺失时会抛出 `FeatureExtractionError`。
