[🇮🇩 Bahasa Indonesia](README_id.md)
# Overview
ICVLPR is a research repository focused on *machine learning* and *computer vision*, developed as part of an undergraduate thesis project. This repository aims to evaluate the performance of *state-of-the-art* [ANPR](https://en.wikipedia.org/wiki/Automatic_number-plate_recognition) models on recognizing commercial vehicle license plates in Indonesia.

# Dataset
The dataset consists of approximately 800 images of commercial vehicle license plates in Indonesia, collected from YouTube videos and direct capture on the streets of Semarang, Indonesia.

The dataset is divided into three sets: `train`, `val`, and `test`, with an 80%, 10%, and 10% split, respectively.

# Usage
## Training
To train the model, make sure you have the following required packages installed:
- PyTorch 2.4.0
- NumPy
- PIL (Python Imaging Library)
- tqdm
- wandb (optional)

To start training, run the following command in your shell:
```shell
python train.py
```

You can also modify the _hyperparameters_ during training. To see the available options, use the `--help` flag:
```shell
python train.py --help
```
