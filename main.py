import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
import time
import copy

np.random.seed(42)
tf.random.set_seed(42)
# Using smaller subsets to keep the ABC cycles from taking all day
N_TRAIN, N_TEST = 10000, 2000
BATCH_SIZE = 128
def prepare_data():
print("Prepping CIFAR-10 data...")
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# Normalize and one-hot encode
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
train_ds = tf.data.Dataset.from_tensor_slices((x_train[:N_TRAIN],
y_train[:N_TRAIN])).batch(BATCH_SIZE).prefetch(2)
test_ds = tf.data.Dataset.from_tensor_slices((x_test[:N_TEST],
y_test[:N_TEST])).batch(BATCH_SIZE).prefetch(2)
return train_ds, test_ds
def get_cnn():
"""Simple 3-block CNN for CIFAR-10 classification."""
model = models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', padding='same',
input_shape=(32, 32, 3)),
layers.BatchNormalization(),
layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
layers.MaxPooling2D(2, 2),
layers.Dropout(0.25),
layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
layers.BatchNormalization(),
layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
layers.MaxPooling2D(2, 2),
layers.Dropout(0.25),
layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
layers.BatchNormalization(),
layers.MaxPooling2D(2, 2),
layers.Dropout(0.4),
layers.Flatten(),
layers.Dense(256, activation='relu'),
layers.Dropout(0.5),
layers.Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
return model
# --- Quantization Helpers ---
def apply_fp8_quant(weights, bits=8):
"""Uniform min-max quantization to simulate 8-bit float precision."""
w_min, w_max = np.min(weights), np.max(weights)
if w_min == w_max:
return weights
levels = (2 ** bits) - 1
scale = (w_max - w_min) / levels
quantized = np.round((weights - w_min) / scale) * scale + w_min
return quantized.astype(np.float32)
def quantize_model(model):
"""Pushes FP8 quantization to all layers with weights."""
for layer in model.layers:
orig_weights = layer.get_weights()
if orig_weights:
quant_weights = [apply_fp8_quant(w) for w in orig_weights]
layer.set_weights(quant_weights)
# --- ABC Evolutionary Logic ---
def get_fitness(model, data):
_, acc = model.evaluate(data, verbose=0)
return float(acc)
def mutate_weights(w_i, w_k, phi, sigma=0.015):
"""
Core ABC mutation logic.
Uses a sigmoid probability to decide which weights get a nudge.
"""
mutated = []
for i, k in zip(w_i, w_k):
velocity = i + phi * (i - k)
prob = 1.0 / (1.0 + np.exp(-np.clip(velocity, -500, 500)))
mask = (np.random.rand(*i.shape) < prob)
noise = np.random.normal(0, sigma, i.shape)
new_w = i + (mask * noise)
mutated.append(new_w.astype(np.float32))
return mutated
# --- Main Execution ---
train_ds, test_ds = prepare_data()
model = get_cnn()
print("\n>>> Phase 1: Pre-training the base model...")
history_pre = model.fit(train_ds, validation_data=test_ds, epochs=25,
verbose=1)
base_acc = get_fitness(model, test_ds)
print(f"\nApplying initial quantization... Baseline: {base_acc:.4f}")
quantize_model(model)
quant_acc = get_fitness(model, test_ds)
print(f"Post-quantization accuracy: {quant_acc:.4f}")
# ABC Hyperparams
NUM_BEES = 10
CYCLES = 8
LIMIT = 3 # Stagnation limit for scouts
print("\n>>> Phase 2: Starting Artificial Bee Colony Optimization...")
current_weights = model.get_weights()
population = [[w + np.random.normal(0, 0.01, w.shape) for w in
current_weights] for
_
in range(NUM_BEES)]
scores = []
# Initial Eval
for i, p in enumerate(population):
model.set_weights(p)
quantize_model(model)
scores.append(get_fitness(model, test_ds))
best_idx = np.argmax(scores)
best_score = scores[best_idx]
best_weights = copy.deepcopy(population[best_idx])
trials = [0] * NUM_BEES
history_abc = []
for c in range(CYCLES):
start_time = time.time()
# 1. Employed Bees
for i in range(NUM_BEES):
k = np.random.choice([idx for idx in range(NUM_BEES) if idx != i])
phi = np.random.uniform(-1, 1)
candidate = mutate_weights(population[i], population[k], phi)
model.set_weights(candidate)
quantize_model(model)
cand_score = get_fitness(model, test_ds)
if cand_score > scores[i]:
population[i], scores[i], trials[i] = model.get_weights(),
cand_score, 0
else:
trials[i] += 1
# 2. Onlooker Bees (Fitness-weighted selection)
norm_scores = np.array(scores) - min(scores) + 1e-9
probs = norm_scores / norm_scores.sum()
for
_
in range(NUM_BEES):
i = np.random.choice(NUM_BEES, p=probs)
k = np.random.choice([idx for idx in range(NUM_BEES) if idx != i])
candidate = mutate_weights(population[i], population[k],
np.random.uniform(-1,1))
model.set_weights(candidate)
quantize_model(model)
cand_score = get_fitness(model, test_ds)
if cand_score > scores[i]:
population[i], scores[i], trials[i] = model.get_weights(),
cand_score, 0
else:
trials[i] += 1
# 3. Scout Bees
for i in range(NUM_BEES):
if trials[i] >= LIMIT:
print(f" [Scout] Bee {i} hit limit. Resetting...")
population[i] = [w + np.random.normal(0, 0.05, w.shape) for w
in best_weights]
model.set_weights(population[i])
quantize_model(model)
scores[i] = get_fitness(model, test_ds)
trials[i] = 0
# Global tracking
if max(scores) > best_score:
best_score = max(scores)
best_weights = copy.deepcopy(population[np.argmax(scores)])
history_abc.append(best_score)
print(f"Cycle {c+1}/{CYCLES} - Best Score: {best_score:.4f} - Time:
{time.time()-start_time:.1f}s")
# --- Wrap up & Viz ---
print("\n>>> Phase 3: Final Polishing (Fine-tuning)...")
model.set_weights(best_weights)
history_fine = model.fit(train_ds, validation_data=test_ds, epochs=5,
verbose=1)
quantize_model(model)
final_acc = get_fitness(model, test_ds)
print("\n" + "="*30)
print(f"Final Accuracy: {final_acc*100:.2f}%")
print("="*30)