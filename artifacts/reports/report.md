# SafeNet evaluation report

Model version `1.0.0+202611171028`  
Generated 2026-11-17 10:28 UTC

Data: CIC-IDS2017 2,520,798 unique flows (train 898,019 / test 756,240); UNSW-NB15 official split (train 175,341 / test 82,332).

## In-dataset

| Dataset | Features | Model | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|---|
| cicids2017 | native | RandomForest | 0.9921 | 0.9993 | 0.9957 | 0.0016 | 0.9999 |
| cicids2017 | native | XGBoost | 0.9951 | 0.9996 | 0.9973 | 0.0010 | 1.0000 |
| cicids2017 | native | Autoencoder | 0.8982 | 0.4694 | 0.6166 | 0.0108 | 0.9691 |
| unswnb15 | native | RandomForest | 0.8434 | 0.9769 | 0.9053 | 0.2222 | 0.9828 |
| unswnb15 | native | XGBoost | 0.8352 | 0.9786 | 0.9012 | 0.2366 | 0.9834 |
| unswnb15 | native | Autoencoder | 0.9458 | 0.3104 | 0.4674 | 0.0218 | 0.8754 |
| cicids2017 | common | RandomForest | 0.9828 | 0.9996 | 0.9911 | 0.0036 | 0.9999 |
| cicids2017 | common | XGBoost | 0.9833 | 0.9997 | 0.9914 | 0.0035 | 1.0000 |
| cicids2017 | common | Autoencoder | 0.9166 | 0.5534 | 0.6901 | 0.0102 | 0.9571 |
| unswnb15 | common | RandomForest | 0.8541 | 0.9587 | 0.9034 | 0.2006 | 0.9777 |
| unswnb15 | common | XGBoost | 0.8515 | 0.9615 | 0.9032 | 0.2054 | 0.9792 |
| unswnb15 | common | Autoencoder | 0.9642 | 0.4420 | 0.6062 | 0.0201 | 0.7743 |

## Cross-dataset (common features, benign vs attack)

| Train | Test | Model | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|---|
| cicids2017 | cicids2017 | RandomForest | 0.9839 | 0.9993 | 0.9915 | 0.0033 | 1.0000 |
| cicids2017 | unswnb15 | RandomForest | 1.0000 | 0.0002 | 0.0005 | 0.0000 | 0.5715 |
| cicids2017 | cicids2017 | XGBoost | 0.9848 | 0.9992 | 0.9919 | 0.0031 | 1.0000 |
| cicids2017 | unswnb15 | XGBoost | 0.2055 | 0.0020 | 0.0039 | 0.0093 | 0.3105 |
| cicids2017 | cicids2017 | Autoencoder | 0.9166 | 0.5534 | 0.6901 | 0.0102 | 0.9571 |
| cicids2017 | unswnb15 | Autoencoder | 0.6807 | 0.5819 | 0.6274 | 0.3345 | 0.7089 |
| unswnb15 | cicids2017 | RandomForest | 0.1819 | 0.6019 | 0.2794 | 0.5501 | 0.5134 |
| unswnb15 | unswnb15 | RandomForest | 0.8790 | 0.9525 | 0.9142 | 0.1607 | 0.9787 |
| unswnb15 | cicids2017 | XGBoost | 0.1888 | 0.5956 | 0.2867 | 0.5202 | 0.5178 |
| unswnb15 | unswnb15 | XGBoost | 0.8530 | 0.9609 | 0.9037 | 0.2029 | 0.9792 |
| unswnb15 | cicids2017 | Autoencoder | 0.2678 | 0.9745 | 0.4202 | 0.5413 | 0.8369 |
| unswnb15 | unswnb15 | Autoencoder | 0.9642 | 0.4420 | 0.6062 | 0.0201 | 0.7743 |
| combined | cicids2017 | RandomForest | 0.9846 | 0.9988 | 0.9916 | 0.0032 | 1.0000 |
| combined | unswnb15 | RandomForest | 0.7749 | 0.9837 | 0.8669 | 0.3501 | 0.9762 |
| combined | cicids2017 | XGBoost | 0.9854 | 0.9984 | 0.9919 | 0.0030 | 1.0000 |
| combined | unswnb15 | XGBoost | 0.7986 | 0.9795 | 0.8798 | 0.3026 | 0.9787 |
| combined | cicids2017 | Autoencoder | 0.8825 | 0.3783 | 0.5296 | 0.0102 | 0.9317 |
| combined | unswnb15 | Autoencoder | 0.9816 | 0.4439 | 0.6114 | 0.0102 | 0.7932 |

## Adversarial robustness (adaptive pad/slow/inject attack)

| Model | Budget | Detection rate |
|---|---|---|
| RandomForest | eps=0.0 | 0.9949 |
| XGBoost | eps=0.0 | 0.9936 |
| XGBoost-AdvTrained | eps=0.0 | 0.9919 |
| Autoencoder | eps=0.0 | 0.4016 |
| RandomForest | eps=0.25 | 0.7947 |
| XGBoost | eps=0.25 | 0.7538 |
| XGBoost-AdvTrained | eps=0.25 | 0.9715 |
| Autoencoder | eps=0.25 | 0.2686 |
| RandomForest | eps=0.5 | 0.7591 |
| XGBoost | eps=0.5 | 0.7403 |
| XGBoost-AdvTrained | eps=0.5 | 0.9761 |
| Autoencoder | eps=0.5 | 0.2302 |
| RandomForest | eps=1.0 | 0.6612 |
| XGBoost | eps=1.0 | 0.6271 |
| XGBoost-AdvTrained | eps=1.0 | 0.9701 |
| Autoencoder | eps=1.0 | 0.2186 |
| RandomForest | eps=2.0 | 0.5457 |
| XGBoost | eps=2.0 | 0.6021 |
| XGBoost-AdvTrained | eps=2.0 | 0.8751 |
| Autoencoder | eps=2.0 | 0.2180 |

## Adversarial training, clean performance

| Model | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| XGBoost | 0.9293 | 0.9935 | 0.9603 | 0.0197 |
| XGBoost-AdvTrained | 0.9382 | 0.9909 | 0.9638 | 0.0170 |
