# ============================================================
# 1. Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import tensorflow as tf
import logging

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, sigmoid

logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ============================================================
# 2. Load and Prepare Data
# ============================================================
from pathlib import Path # allows file location checking
folder = Path(__file__).parent # Makes sure to check for finaldata.csv in the same folder as this script
df = pd.read_csv(folder / "data.csv")

X = df[["speed", "elevation degree", "spin rpm"]].to_numpy(dtype=np.float32)
Y = df["result_scored"].to_numpy(dtype=np.float32)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Unique y:", np.unique(Y))

# ============================================================
# 3. Train-Test Split
# ============================================================
np.random.seed(42)
m = len(X)
perm = np.random.permutation(m)

test_size = int(0.2 * m)
test_idx = perm[:test_size]
train_idx = perm[test_size:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

print("Train:", X_train.shape, Y_train.shape)
print("Test: ", X_test.shape, Y_test.shape)


# ============================================================
# 4. Create the Neural Network
# ============================================================

tf.random.set_seed(1234)
model = Sequential(
    [
        Dense(64, activation=relu, name="L1"),
        Dense(32, activation=relu, name="L2"),
        Dense(1, activation=sigmoid, name="L3"),
    ],  name="OutcomePredictor"
)

# Compile the model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=["accuracy"]
)

# ============================================================
# 5. Train the Model
# ============================================================

from tensorflow.keras.callbacks import EarlyStopping
early = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)

model.fit(
    X_train, Y_train,
    epochs=1000,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print("TEST loss:", test_loss)
print("TEST accuracy:", test_acc)

# ============================================================
# 7. Model Summary
# ============================================================

model.summary()

#saving the model
model.save("outcome_model.keras") 