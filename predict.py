import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path


folder = Path(__file__).parent

time = load_model(folder / "time_model.keras")
score = load_model(folder / "outcome_model.keras")

new_shot = np.array([[26.8, 15.87, 300.0]], dtype=np.float32)
print("Inputs:" + "\ninitial speed: " + str(new_shot[0][0]) + "\nangle: " + str(new_shot[0][1]) + "\nspin rpm: " + str(new_shot[0][2]))
print("Chance of Scoring:", score.predict(new_shot)*100, "%")
if score.predict(new_shot)[0][0] >= 0.5:
    print(">> GOAL")
    print("Time prediction:", time.predict(new_shot))
else:
    print(">> MISS")

time.layers[0].get_weights()