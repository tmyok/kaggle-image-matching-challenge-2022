# Image Matching Challenge 2022
This repository contains code used achieve 10th place in the Image Matching Challenge 2022 competition which was hosted on kaggle (https://www.kaggle.com/competitions/image-matching-challenge-2022)

Our solution is described in the following link.  
Kaggle Discussion : https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/328903  
Konica Minolta FORXAI engineers blog (Written in Japanese) : https://forxai.konicaminolta.com/blog/017

# Usage

## Prepare
1. Clone this repository
   ```
   git clone --recursive https://github.com/tmyok/kaggle-image-matching-challenge-2022.git
   ```
2. Download datasets in input dir; refer to [input/README.md](input/README.md).
3. Run a docker container (if necessary).
   ```
   sh docker_container.sh
   ```

## Inference

```
cd working
python3 exec_test_data.py
```

## Validation

```
cd working
python3 exec_validation_data_pipeline.py
```

## Hyperparameter search using Hydra Optuna Sweeper
   ```
   cd working
   sh tuning_sample.sh
   ```
