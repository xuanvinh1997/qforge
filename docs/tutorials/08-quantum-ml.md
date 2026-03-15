# Tutorial 8: Quantum Machine Learning

This tutorial covers quantum feature encoding, quantum kernel methods, and
integration with scikit-learn for classification tasks.

## 1. Data Encoding Strategies

Encoding classical data into quantum states is the first step in any QML
pipeline. Qforge provides several strategies, each with different
expressibility and entanglement properties.

### Amplitude Encoding

Encodes a normalized vector of length 2^n into the amplitudes of n qubits:

```python
from Qforge.circuit import Qubit
from Qforge.encodings import amplitude_encode
import numpy as np

# Encode a 4-element vector into 2 qubits
data = np.array([0.5, 0.5, 0.5, 0.5])  # Must be normalized
qc = Qubit(n_qubits=2)
amplitude_encode(qc, data)

print(qc.wavefunction.amplitude)
# [0.5+0j, 0.5+0j, 0.5+0j, 0.5+0j]
```

> **Note:** Amplitude encoding is exponentially compact (n qubits encode 2^n
> features) but requires O(2^n) gates to prepare. Best for small feature
> vectors or when qubit count is limited.

### Angle Encoding (YZ-CX)

Maps each feature to a rotation angle on a dedicated qubit:

```python
from Qforge.encodings import yz_cx_encode

# 4 features -> 4 qubits
data = np.array([0.3, 1.2, 0.7, 2.1])
qc = Qubit(n_qubits=4)
yz_cx_encode(qc, data)
```

### ZZ Feature Map

Creates entanglement between qubits based on pairwise feature products,
similar to Qiskit's ZZFeatureMap:

```python
from Qforge.encodings import zz_feature_map_encode

data = np.array([0.5, 1.0, 0.8])
qc = Qubit(n_qubits=3)
zz_feature_map_encode(qc, data, reps=2)
```

> **Tip:** The ZZ feature map produces classically hard-to-simulate kernel
> functions, making it a strong candidate for quantum advantage in kernel
> methods. Use `reps=2` or higher for better expressibility.

## 2. Quantum Kernel Methods

Quantum kernels measure the similarity between data points in Hilbert space.
Qforge provides two approaches: the swap test and projected quantum kernels.

### Swap Test

The swap test computes the overlap |<psi|phi>|^2 between two quantum states:

```python
from Qforge.kernels import swap_test

x1 = np.array([0.3, 1.2, 0.7, 2.1])
x2 = np.array([0.5, 1.0, 0.8, 1.9])

overlap = swap_test(x1, x2, encoding='yz_cx')
print(f"Kernel value: {overlap:.4f}")
```

### Projected Quantum Kernel

The `ProjectedQuantumKernel` computes kernel matrices compatible with
scikit-learn estimators:

```python
from Qforge.kernels import ProjectedQuantumKernel

# Create a quantum kernel with ZZ feature map encoding
qkernel = ProjectedQuantumKernel(
    n_qubits=4,
    encoding='zz_feature_map',
    reps=2,
    projection='xyz',  # Project onto X, Y, Z expectations
)

# Compute the kernel matrix for a dataset
X_train = np.random.uniform(0, np.pi, size=(20, 4))
K = qkernel.kernel_matrix(X_train)
print(f"Kernel matrix shape: {K.shape}")  # (20, 20)
```

## 3. Classification with Quantum SVM

Combine the quantum kernel with scikit-learn's SVC:

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Qforge.kernels import ProjectedQuantumKernel
import numpy as np

# Generate a 2D classification dataset
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create quantum kernel
qkernel = ProjectedQuantumKernel(
    n_qubits=2,
    encoding='zz_feature_map',
    reps=2,
)

# Compute kernel matrices
K_train = qkernel.kernel_matrix(X_train)
K_test = qkernel.kernel_matrix(X_test, X_train)  # Cross-kernel

# Train SVM with precomputed kernel
svm = SVC(kernel='precomputed')
svm.fit(K_train, y_train)

# Evaluate
y_pred = svm.predict(K_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.2%}")
```

## 4. Visualizing the Decision Boundary

```python
import matplotlib.pyplot as plt

# Create a mesh grid for the decision boundary
h = 0.1
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Compute kernel between grid and training points
K_grid = qkernel.kernel_matrix(grid_points, X_train)
Z = svm.predict(K_grid).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdBu',
            edgecolors='k', s=60)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Quantum SVM (accuracy: {acc:.2%})')
plt.colorbar()
plt.tight_layout()
plt.savefig('quantum_svm.png', dpi=150)
plt.show()
```

## 5. Comparing Encoding Strategies

Different encodings suit different datasets. Compare them systematically:

```python
from sklearn.model_selection import cross_val_score

encodings = ['amplitude', 'yz_cx', 'zz_feature_map']
results = {}

for enc in encodings:
    qk = ProjectedQuantumKernel(
        n_qubits=2,
        encoding=enc,
        reps=2,
    )
    K_full = qk.kernel_matrix(X)

    svm = SVC(kernel='precomputed')
    scores = cross_val_score(svm, K_full, y, cv=5)
    results[enc] = scores.mean()
    print(f"{enc:20s}: accuracy = {scores.mean():.2%} (+/- {scores.std():.2%})")
```

> **Warning:** Quantum kernel computation scales as O(n^2) in the number of
> data points (each pair requires a circuit evaluation). For datasets larger
> than a few hundred points, consider subsampling or using Nystrom
> approximation.

## 6. Variational Quantum Classifier

For a trainable approach, combine a parameterized circuit with a classical
optimizer:

```python
from Qforge.circuit import Qubit
from Qforge.gates import H, CNOT, RY, RZ
from Qforge.measurement import pauli_expectation
from Qforge.algo import Hamiltonian, parameter_shift
import numpy as np

def variational_classifier(features, params, n_layers=2):
    """Parameterized circuit for binary classification."""
    n_qubits = len(features)
    qc = Qubit(n_qubits=n_qubits)

    # Encode features
    for i in range(n_qubits):
        RY(qc, target=i, theta=features[i])

    # Variational layers
    idx = 0
    for layer in range(n_layers):
        for i in range(n_qubits):
            RY(qc, target=i, theta=params[idx])
            RZ(qc, target=i, theta=params[idx + 1])
            idx += 2
        for i in range(n_qubits - 1):
            CNOT(qc, control=i, target=i + 1)

    return qc

def predict(features, params):
    """Return +1 or -1 based on Z expectation of qubit 0."""
    qc = variational_classifier(features, params)
    obs = Hamiltonian(coeffs=[1.0], terms=[[('Z', 0)]])
    return pauli_expectation(qc, obs)

def cost(params, X, y):
    """Mean squared error cost function."""
    total = 0.0
    for xi, yi in zip(X, y):
        pred = predict(xi, params)
        total += (pred - yi) ** 2
    return total / len(X)

# Training loop
n_qubits = 2
n_layers = 2
n_params = n_qubits * n_layers * 2
params = np.random.uniform(-np.pi, np.pi, size=n_params)
y_signed = 2 * y_train - 1  # Convert {0, 1} -> {-1, +1}

lr = 0.1
for epoch in range(50):
    cost_fn = lambda p: cost(p, X_train, y_signed)
    grads = parameter_shift(cost_fn, params)
    params -= lr * grads
    if epoch % 10 == 0:
        c = cost_fn(params)
        print(f"Epoch {epoch:3d}: cost = {c:.4f}")
```

## Summary

| Concept | API |
|---------|-----|
| Amplitude encoding | `amplitude_encode(qc, data)` |
| Angle encoding | `yz_cx_encode(qc, data)` |
| ZZ feature map | `zz_feature_map_encode(qc, data, reps=2)` |
| Swap test | `swap_test(x1, x2, encoding='...')` |
| Quantum kernel | `ProjectedQuantumKernel(n_qubits, encoding, reps)` |
| Kernel matrix | `qkernel.kernel_matrix(X)` or `qkernel.kernel_matrix(X, Y)` |

---

Next: [Tutorial 9: Noise and Error Mitigation](09-noise-and-mitigation.md)
