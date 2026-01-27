# ============================================================
# Land Use / Land Cover Change Analysis in the Niger Delta
# Using Multi-Temporal Satellite Data and ML (Synthetic Data)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------
# 1. Synthetic Multi-Temporal Satellite Data Generator
# ------------------------------------------------------------

def generate_multispectral_data(size=120, year="t1"):
    blue = np.random.normal(0.12, 0.03, (size, size))
    green = np.random.normal(0.28, 0.05, (size, size))
    red = np.random.normal(0.22, 0.04, (size, size))
    nir = np.random.normal(0.62, 0.07, (size, size))

    if year == "t2":
        cx, cy = np.random.randint(30, size - 30, 2)
        radius = np.random.randint(20, 35)
        y, x = np.ogrid[:size, :size]
        urban = (x - cx)**2 + (y - cy)**2 <= radius**2

        nir[urban] *= 0.35
        red[urban] *= 1.5
        green[urban] *= 1.3

    return blue, green, red, nir

# ------------------------------------------------------------
# 2. Spectral Indices
# ------------------------------------------------------------

def compute_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-6)

def compute_ndwi(green, nir):
    return (green - nir) / (green + nir + 1e-6)

# ------------------------------------------------------------
# 3. Dataset Creation
# ------------------------------------------------------------

def create_dataset(samples=120):
    X, y = [], []

    for _ in range(samples):
        blue, green, red, nir = generate_multispectral_data()
        ndvi = compute_ndvi(nir, red)
        ndwi = compute_ndwi(green, nir)

        for i in range(ndvi.shape[0]):
            for j in range(ndvi.shape[1]):
                X.append([ndvi[i, j], ndwi[i, j], blue[i, j], nir[i, j]])

                if ndvi[i, j] > 0.45:
                    y.append(0)      # Vegetation
                elif ndwi[i, j] > 0.1:
                    y.append(1)      # Water
                else:
                    y.append(2)      # Built-up

    return np.array(X), np.array(y)

X, y = create_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ------------------------------------------------------------
# 4. ML Model
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=14,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 5. Evaluation
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("LULC Classification Report")
print(classification_report(
    y_test, y_pred,
    target_names=["Vegetation", "Water", "Built-up"]
))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------
# 6. LULC Mapping & Change Detection
# ------------------------------------------------------------

def generate_lulc_map(year="t1"):
    blue, green, red, nir = generate_multispectral_data(year=year)
    ndvi = compute_ndvi(nir, red)
    ndwi = compute_ndwi(green, nir)

    stack = np.stack([ndvi, ndwi, blue, nir], axis=-1)
    reshaped = stack.reshape(-1, 4)

    pred = model.predict(reshaped)
    return pred.reshape(ndvi.shape)

lulc_t1 = generate_lulc_map("t1")
lulc_t2 = generate_lulc_map("t2")
change_map = lulc_t2 - lulc_t1

# ------------------------------------------------------------
# 7. Visualization
# ------------------------------------------------------------

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.title("LULC Map - Time 1")
plt.imshow(lulc_t1, cmap="tab10")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("LULC Map - Time 2")
plt.imshow(lulc_t2, cmap="tab10")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("LULC Change Detection")
plt.imshow(change_map, cmap="coolwarm")
plt.axis("off")

plt.tight_layout()
plt.show()
