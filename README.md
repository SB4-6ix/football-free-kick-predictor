
# Football Free-Kick Predictor
Predicting football free-kick scoring outcomes and goal-crossing times using physics-based simulation and neural networks built with Python and TensorFlow.

### 🌐 Live Interactive Application
**[Try the Free-Kick Predictor](https://football-free-kick-predictor.onrender.com)**

Adjust the initial ball speed, elevation angle, and spin rate to generate predictions using two trained neural networks.

> **Note:** The application is hosted on Render's free tier. The first visit after inactivity may take approximately a minute to load.

## Project Overview
This project uses machine learning to predict whether a football free kick will result in a goal and, if successful, estimate the time taken for the ball to reach the goal.

Two artificial neural networks were developed using a dataset of **5,850 simulated free kicks**, generated through a physics-based simulation accounting for gravity, aerodynamic drag, and the Magnus effect.

The project combines numerical physics simulation, synthetic dataset generation, supervised machine learning, and predictive modelling using Python and TensorFlow.

Both neural networks take three input features:
- **Initial Speed:** Ball velocity in metres per second (m/s)
- **Elevation Angle:** Vertical launch angle in degrees (°)
- **Spin Rate:** Ball rotation in revolutions per minute (RPM)

The classification model predicts the scoring outcome, while the regression model estimates goal-crossing time for successful shots. An interactive Streamlit application allows users to adjust the three parameters and generate predictions without modifying Python code.

> **Note:** The models are trained on synthetic data under fixed simulation conditions. Their predictions should not be interpreted as validated real-world football outcomes.

---

## 1. Project Workflow
The project follows five main stages.

### Stage 1 — Physics Simulation
**File:** `free_kick_simulator.py`

Models football motion using gravitational force, aerodynamic drag, and the Magnus effect. The simulator uses fourth-order Runge–Kutta (RK4) numerical integration to calculate trajectories and determine whether a shot crosses the goal plane within the goal boundaries.

### Stage 2 — Dataset Generation
**File:** `generate_data.py`  
**Output:** `data.csv`

Runs 5,850 free-kick simulations using different combinations of initial speed, elevation angle, and spin rate. Each simulation records the initial conditions, scoring outcome, and goal-crossing time if applicable.

### Stage 3 — Neural Network Training
**Classification:** `train_outcome_model.py` → `outcome_model.keras`  
**Regression:** `train_time_model.py` → `time_model.keras`

The classification model predicts whether a shot scores, while the regression model estimates goal-crossing time for successful free kicks.

### Stage 4 — Prediction
**File:** `predict.py`

Loads both saved neural networks and generates predictions using input parameters specified directly in Python.

### Stage 5 — Interactive Streamlit Application
**File:** `app.py`  
**Theme configuration:** `.streamlit/config.toml`

Provides a browser-based interface with interactive sliders for initial speed, elevation angle, and spin rate.

The application displays:
- Model-estimated scoring probability
- Predicted goal or miss
- Estimated goal-crossing time for predicted goals

The Streamlit application has been tested locally. A public deployment is not yet available.

---

## 2. Project File Structure
All project files are stored in the repository's main directory unless otherwise indicated.

```text
football-free-kick-predictor/
├── .streamlit/
│   └── config.toml           # Streamlit appearance and theme
├── app.py                    # Interactive Streamlit application
├── free_kick_simulator.py    # Physics simulation engine
├── generate_data.py          # Synthetic dataset generator
├── data.csv                  # 5,850 simulated free kicks
├── train_outcome_model.py    # Classification model training
├── train_time_model.py       # Regression model training
├── outcome_model.keras       # Saved classification neural network
├── time_model.keras          # Saved regression neural network
├── predict.py                # Python prediction script
├── requirements.txt          # Required Python libraries
└── README.md                 # Project documentation
```

### Recommended Reading Order
For someone interested in understanding the project, the recommended order is:

1. **`free_kick_simulator.py`** — Understand the physics equations and numerical simulation.
2. **`generate_data.py`** — Understand how simulations are used to generate training data.
3. **`data.csv`** — Examine the simulated inputs and outcomes.
4. **`train_outcome_model.py`** — Understand classification network training.
5. **`train_time_model.py`** — Understand regression network training.
6. **`predict.py`** — See how the saved models generate predictions.
7. **`app.py`** — Explore the interactive interface connected to both models.
8. **`.streamlit/config.toml`** — Review the website's theme settings.

The `.keras` files contain the saved neural networks and can be loaded without retraining.

---

## 3. Physics Simulation
**File:** `free_kick_simulator.py`

The physics engine simulates football motion in three-dimensional space using three main forces:
- **Gravity:** Accelerates the ball downward.
- **Aerodynamic Drag:** Opposes the ball's motion relative to the surrounding air.
- **Magnus Force:** Produces lift and sideways movement due to ball rotation.

The simulation uses **fourth-order Runge–Kutta numerical integration (RK4)** with a time step of 0.002 seconds to approximate the ball's position and velocity throughout its trajectory.

A shot is classified as a goal when the ball crosses the goal plane within the permitted width and height, accounting for the ball's radius.

The simulator returns trajectory data, timestamps, and a scoring result. Only the relevant input and output variables are retained in the final machine-learning dataset.

---

## 4. Dataset Generation
**File:** `generate_data.py`  
**Dataset:** `data.csv`

The dataset was generated by varying three free-kick parameters:

| Parameter | Range | Increment | Number of Values |
|---|---|---|---|
| Initial Speed | 25–32 m/s | 0.5 m/s | 15 |
| Elevation Angle | 14–19° | 0.2° | 26 |
| Spin Rate | 0–350 RPM | 25 RPM | 15 |

**Total simulations: 15 × 26 × 15 = 5,850**

### Dataset Columns
| Column | Description |
|---|---|
| `speed` | Initial ball speed (m/s) |
| `elevation degree` | Launch angle (degrees) |
| `spin rpm` | Ball spin rate (RPM) |
| `result_time` | Goal-crossing time in seconds, if scored |
| `result_scored` | Whether the free kick scored (`True` or `False`) |
| `result_point` | Ball-centre coordinates at the goal plane, if scored |

For missed shots, `result_time` and `result_point` are left blank because no successful goal crossing occurred.

### Input and Output Variables
Both neural networks use the same input vector:

**X = [speed, elevation angle, spin rate]**

The target variables differ:
- **Classification:** `result_scored` (goal or miss)
- **Regression:** `result_time` (seconds, for successful shots only)

---

## 5. Neural Network Models
Both models were implemented using TensorFlow/Keras.

### Model A — Free-Kick Outcome Classification
**Training script:** `train_outcome_model.py`  
**Saved model:** `outcome_model.keras`

This model predicts whether a free kick will result in a goal or miss.

#### Architecture
- Input: 3 features
- Hidden Layer 1: 64 neurons, ReLU
- Hidden Layer 2: 32 neurons, ReLU
- Output Layer: 1 neuron, Sigmoid

#### Training Configuration
| Setting | Value |
|---|---|
| Loss Function | Binary Cross-Entropy |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Maximum Epochs | 1,000 |
| Batch Size | 128 |
| Test Split | 20% |
| Validation Split | 20% of training portion |
| Early Stopping Patience | 50 epochs |

The sigmoid output ranges from 0 to 1 and is presented as a **model-estimated scoring probability** for the fixed simulation setup. Probability calibration has not been independently evaluated.

A threshold of 0.5 is used to classify each shot:
- **Output ≥ 0.5:** Predicted goal
- **Output < 0.5:** Predicted miss

The model's training script evaluates binary cross-entropy loss and classification accuracy.

### Model B — Goal-Crossing Time Regression
**Training script:** `train_time_model.py`  
**Saved model:** `time_model.keras`

This model estimates the time required for a successful free kick to reach the goal plane. Only successful shots are included during regression training.

#### Architecture
- Input: 3 features
- Hidden Layer 1: 64 neurons, ReLU
- Hidden Layer 2: 32 neurons, ReLU
- Output Layer: 1 neuron, Linear

#### Training Configuration
| Setting | Value |
|---|---|
| Loss Function | Mean Squared Error |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Maximum Epochs | 500 |
| Batch Size | 64 |
| Test Split | 20% |
| Validation Split | 20% of training portion |
| Early Stopping Patience | 50 epochs |

### Regression Model Performance
Previously recorded test-set results:

| Metric | Result |
|---|---|
| Mean Squared Error (MSE) | 0.0011947 s² |
| Root Mean Squared Error (RMSE) | 0.0345645 s |
| Mean Absolute Error (MAE) | 0.0275935 s |

The MAE indicates that the model's predictions differ from simulated goal-crossing times by approximately **27.6 milliseconds on average** for the test observations.

These values describe performance on held-out synthetic observations, not real-world football data.

---

## 6. How to Run the Project

### Requirements
The project uses Python and the following libraries:
- NumPy
- Pandas
- TensorFlow/Keras
- Matplotlib
- Streamlit

Python 3.11 was used for the working local Streamlit environment.

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

### Option A — Run the Interactive Streamlit App (Recommended)
**File:** `app.py`

The interactive application allows users to generate predictions without modifying Python code.

From the repository directory, run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

The app provides three sliders:
- Initial ball speed: 25–32 m/s
- Elevation angle: 14–19°
- Spin rate: 0–350 RPM

Click **Predict Free Kick** to generate predictions.

The application displays the model-estimated scoring probability and goal/miss classification. For predicted goals, the regression network also estimates goal-crossing time.

The interface uses Streamlit with additional CSS styling and a black-and-gold theme configured through `.streamlit/config.toml`.

**The application currently runs locally and has not yet been publicly deployed.**

### Option B — Run a Prediction Using Python
**File:** `predict.py`

The repository includes two pretrained neural networks, so retraining is not necessary.

Open `predict.py` and locate:

```python
new_shot = np.array([[26.8, 15.87, 300.0]], dtype=np.float32)
```

The input order is:

`[speed (m/s), elevation angle (degrees), spin (RPM)]`

Modify these values to test another free kick.

Run:

```bash
python predict.py
```

The program loads `outcome_model.keras` and `time_model.keras`, displays the model's estimated scoring output, and predicts goal-crossing time when the shot is classified as a goal.

### Regenerate the Dataset (Optional)
**File:** `generate_data.py`

Run:

```bash
python generate_data.py
```

This executes the configured physics simulations and saves their results to `data.csv`.

**Warning:** Running this script overwrites the existing dataset.

### Retrain the Models (Optional)
Train the classification model:

```bash
python train_outcome_model.py
```

Train the regression model:

```bash
python train_time_model.py
```

Both scripts load `data.csv`, perform training and evaluation, and save their trained models.

Retraining may overwrite existing `.keras` files. Ensure the classification training script saves to `outcome_model.keras`, matching the filenames used by `app.py` and `predict.py`.

---

## 7. Assumptions and Limitations
This project uses a simplified physics model and synthetic training data rather than recorded professional football matches.

### Fixed Simulation Conditions
| Parameter | Value |
|---|---|
| Distance to Goal | 22.0 m |
| Initial Lateral Position | 0 m (aligned with goal centre) |
| Initial Ball Height | 0.23 m |
| Kick Azimuth | 0.5° |
| Spin-Axis Azimuth | 60° |
| Spin-Axis Elevation | 8° |
| Wind Velocity | (2.0, −1.0, 0.0) m/s |
| Air Density | 1.20 kg/m³ |
| Drag Coefficient | 0.28 |
| Ball Mass | 0.43 kg |
| Ball Radius | 0.11 m |

The free kick originates on the central axis of the goal, with its horizontal launch angle fixed at 0.5°.

Only initial speed, elevation angle, and spin rate are varied across the dataset.

### Model Limitations
- **Synthetic Data:** The neural networks learn from a mathematical simulation rather than real match observations.
- **Fixed Location:** The dataset does not vary free-kick distance or lateral shooting position.
- **Fixed Atmospheric Conditions:** Wind, air density, and drag coefficient remain constant.
- **Limited Input Ranges:** Predictions outside the simulated parameter ranges have not been validated.
- **No Defensive Interactions:** Defensive walls and goalkeepers are not simulated.
- **Simplified Aerodynamics:** Effects such as changing aerodynamic coefficients and spin decay are not fully represented.
- **Regression Scope:** Travel-time predictions apply only to successful free kicks.
- **Generalization:** Random train-test splitting measures performance on held-out combinations from the same simulated environment, not entirely new physical conditions.
- **Probability Calibration:** The classification model's sigmoid outputs have not been independently tested for calibration.

Therefore, the models should be interpreted as experimental prediction tools for the defined simulation environment.

---

## 8. Future Improvements
Potential future developments include:
- Expanding the dataset to include different shooting distances and lateral positions.
- Varying kick azimuth and spin-axis direction.
- Introducing more diverse wind and environmental conditions.
- Comparing predictions with real free-kick trajectory data.
- Evaluating classification performance using precision, recall, F1-score, and confusion matrices.
- Testing alternative machine-learning algorithms and feature normalization.
- Deploying the existing Streamlit interface publicly.
- Adding interactive three-dimensional free-kick trajectory visualizations.

---

## Conclusion
This project demonstrates how physics-based numerical simulation can be combined with supervised machine learning to model and predict outcomes of a physical system.

Through dataset generation, neural network training, evaluation, prediction, and a locally tested interactive Streamlit application, the project explores how initial ball speed, elevation angle, and spin rate influence simulated free-kick scoring outcomes and goal-crossing times.
