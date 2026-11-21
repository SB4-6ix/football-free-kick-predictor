# ============================================================
# 1. Import Libraries
# ============================================================

import pandas as pd
import numpy as np
import tensorflow as tf
import logging

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu

logging.getLogger("tensorflow").setLevel(logging.ERROR) #Controls which informational messages, warnings, and errors appear in your terminal


# ============================================================
# 2. Load and Prepare Data
# ============================================================

from pathlib import Path # allows file location checking
folder = Path(__file__).parent # Makes sure to check for finaldata.csv in the same folder as this script
df = pd.read_csv(folder / "data.csv")

# Filter data to only successful free kicks
df = df[(df["result_scored"] == 1)].copy()

X = df[["speed", "elevation degree", "spin rpm"]].to_numpy(dtype=np.float32)
Y = df["result_time"].to_numpy(dtype=np.float32)

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# ============================================================
# 3. Train-Test Split
# ============================================================

np.random.seed(42) # choose a random set of training and testing data
m = len(X)
perm = np.random.permutation(m)

test_size = int(0.2 * m) # 20% of the data will be used for testing
test_idx = perm[:test_size]
train_idx = perm[test_size:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

print("Train:", X_train.shape, Y_train.shape)
print("Test: ", X_test.shape, Y_test.shape)

print(len(X_train), len(Y_train))

# ============================================================
# 4. Create the Neural Network
# ============================================================

tf.random.set_seed(1234)
model = Sequential(
    [
        Dense(64, activation=relu, name="L1"),
        Dense(32, activation=relu, name="L2"),
        Dense(1, activation=linear, name="L3"),
    ],  name="TimePredictor",
)

# Compile the model
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(0.0001),
)

# ============================================================
# 5. Train the Model
# ============================================================

from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True) # Stops prohram if validation loss does not improve for 50 epochs, and restores the best weights

history = model.fit(
    X_train, Y_train,
    epochs=500,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early],
    verbose=1
)


# ============================================================
# 6. Evaluating the Model using MSE, RMSE, and MAE
# ============================================================

pred = model.predict(X_test).reshape(-1)
mse = np.mean((pred - Y_test)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(pred - Y_test))

print("TEST MSE:", mse)
print("TEST RMSE:", rmse)
print("TEST MAE:", mae)


# ============================================================
# 7. Model Summary
# ============================================================

model.summary() #prints summary of the model architecture

model.save(folder / "time_model.keras") # saves the model weights


# ============================================================
# 8. Training and Validation Loss Graphs
# ============================================================

# Loss Graph (Log Scale)
import matplotlib.pyplot as plt

# Get loss values
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(9, 5))

plt.plot(epochs, train_loss, label="Training Loss", linewidth=2)
plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)

# Use log scale to handle large early losses
plt.yscale("log")

plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss (MSE - Log Scale)", fontsize=12)
plt.title("Training and Validation Loss vs Epochs", fontsize=14)

plt.legend(fontsize=11)
plt.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()