"""Example showing how an external application calls this package."""

from per_epoch_features import extract_features, save_features


features = extract_features(
    data_folder=r"D:\physionet2026\training_set",
    bids_folder="sub-I0002150005420",
    site_id="I0002",
    session_id=1,
)

print("X_seq:", features["X_seq"].shape)
print("X_ecg:", features["X_ecg"].shape)
print("x_static:", features["x_static"].shape)
print("mask:", features["mask"].shape)

save_features(features, "record_features.npz")
