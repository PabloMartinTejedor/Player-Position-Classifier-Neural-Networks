# ⚽ Player Position Classification Based on Performance Statistics

This project focuses on classifying the ideal playing position of a football player based on their statistical performance. Our goal is to determine, using neural networks, which role a player best fits into according to their match-related attributes (tackles, passes, recoveries, etc.).

This classification task is inspired by real-world scenarios such as:

- **Position Reassignment** — Identifying players who might perform better in a different role.
- **Versatility Detection** — Recognizing hybrid or multi-role players.
- **Scouting Optimization** — Suggesting ideal profiles for recruitment.
- **Video Game Integration** — Enabling team auto-generation and improving realism through AI.

The entire pipeline is implemented from scratch using PyTorch and follows a structured machine learning approach: preprocessing, model definition, training, evaluation and comparison.

## 📚 Dataset Description

The dataset used contains statistics from professional football players, such as number of blocks, tackles, progressive passes, goals, and carries, along with the actual position each player performed in. These features are used to classify the ideal position of the player.

You can access the dataset from the following link:  
[🔗 Player Statistics Dataset](https://www.kaggle.com/datasets/vivovinco/20212022-football-team-stats)

## ⚖️ Preprocessing Steps

The preprocessing pipeline includes the following steps:

1. **Feature Selection**: We filtered the dataset to include only the most relevant attributes for player behavior and performance, including the target column `Pos` (position).
2. **Label Encoding**: The target variable `Pos` was categorical (e.g., 'MF', 'DF', etc.), so it was converted to numeric labels using `LabelEncoder`.
3. **Normalization**: All input features were scaled to a range [0, 1] using `MinMaxScaler` to ensure uniform influence during training.
4. **Train/Test Split**: The data was split into 80% for training and 20% for testing. This allows us to train the model on one part of the data and evaluate its ability to generalize to unseen examples.
5. **Tensor Conversion**: The training and test data were converted to PyTorch tensors, with the appropriate data types (`float32` for features and `int64` for targets).
6. **Batching**: We organized the data into batches of 32 using `DataLoader` for efficient training.

## 🧬 Models Trained

We trained and evaluated the following fully connected (dense) neural network architectures:

| Model | Architecture Details                              | Accuracy (%) |
|-------|----------------------------------------------------|--------------|
| 1     | Linear Model (0 Hidden Layers)                     | 32.65        |
| 2     | 3 Hidden Layers + ReLU                             | 65.47        |
| 3     | 3 Hidden Layers + Tanh                             | 66.50        |
| 4     | 3 Hidden Layers + ReLU + Batch Normalization       | 68.89        |
| 5     | 3 Hidden Layers + ReLU + Dropout (p=0.2)           | 62.74        |
| 6     | 3 Hidden Layers + ReLU + Momentum                  | 67.86        |
| 7     | 3 Hidden Layers + ReLU + BatchNorm + Momentum      | 68.03        |

We used accuracy as the evaluation metric since the dataset is balanced across all position categories.

## 📊 Results & Observations

- Adding hidden layers and non-linear activations significantly improved performance over the simple linear model.
- Tanh slightly outperformed ReLU in our tests.
- Batch Normalization yielded the best gains in both training speed and final accuracy.
- Dropout decreased accuracy in this setting, though p=0.2 performed better than p=0.3.
- Momentum accelerated convergence and stabilized training, especially when combined with BatchNorm.
- The error curves showed consistent improvement across 100 epochs for most models.

## ⚡ Execution Instructions

To run this project step by step in Google Colab:

1. Download the dataset from the following Kaggle link:  
   [🔗 2021/2022 Football Team Stats](https://www.kaggle.com/datasets/vivovinco/20212022-football-team-stats)

2. Rename the dataset file to:  
   `Dataset - Proyecto Redes Neuronales.csv`  
   (This name is required for the code to recognize the file correctly.)

3. Upload the dataset manually by executing the following code in the first cell:

   ```python
   from google.colab import files
   uploaded = files.upload()
