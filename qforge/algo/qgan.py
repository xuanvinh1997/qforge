# -*- coding: utf-8 -*-
# author: vinhpx
"""QGAN — Quantum Generative Adversarial Network.

A quantum generator (parameterized circuit) learns to produce quantum
states whose measurement statistics match a target probability
distribution.  The discriminator is a classical neural network (single
hidden layer) trained to distinguish real from generated samples.

Reference: Dallaire-Demers & Killoran — Physical Review A 98, 012324 (2018).

Example::

    from qforge.algo import QGAN
    import numpy as np

    # Target: bimodal distribution
    rng = np.random.default_rng(0)
    real_data = np.concatenate([rng.normal(-2, 0.5, 100),
                                 rng.normal(2, 0.5, 100)])

    qgan = QGAN(n_qubits=3, n_layers=4, n_bins=8)
    g_hist, d_hist = qgan.train(real_data, steps=80)
    generated = qgan.sample(500)
"""
from __future__ import annotations
import numpy as np
from qforge.circuit import Qubit
from qforge.gates import H, RY, RZ, CNOT
from qforge.measurement import measure_one
from qforge.algo.gradient import parameter_shift
from qforge.algo.optimizers import Adam


class QGAN:
    """Quantum Generative Adversarial Network.

    The generator is a parameterized quantum circuit whose output
    probability distribution (Born probabilities) is trained to match
    a target data distribution.  The discriminator is a small classical
    network.

    Args:
        n_qubits:  Number of qubits (generator output has 2^n_qubits bins).
        n_layers:  Number of variational layers in the generator.
        n_bins:    Number of histogram bins for discretizing real data.
                   Automatically set to ``2**n_qubits`` if ``None``.
        backend:   qforge backend string.
    """

    def __init__(
        self,
        n_qubits: int = 3,
        n_layers: int = 4,
        n_bins: int | None = None,
        backend: str = 'auto',
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_bins = n_bins if n_bins is not None else 2 ** n_qubits
        self.backend = backend
        self.dim = 2 ** n_qubits

        # Generator parameters: n_layers * n_qubits * 2 (RY + RZ)
        self.n_gen_params = n_layers * n_qubits * 2
        self.gen_params = None

        # Discriminator: simple 1-hidden-layer MLP
        # Input: n_bins, Hidden: n_bins, Output: 1
        self._disc_params = None

    # ------------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------------

    def _generator_circuit(self, params: np.ndarray) -> np.ndarray:
        """Run the generator circuit, return Born probabilities."""
        wf = Qubit(self.n_qubits, backend=self.backend)
        # Initial superposition
        for q in range(self.n_qubits):
            H(wf, q)

        idx = 0
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                RY(wf, q, params[idx]); idx += 1
                RZ(wf, q, params[idx]); idx += 1
            for q in range(self.n_qubits - 1):
                CNOT(wf, q, q + 1)

        # Born probabilities
        probs = np.abs(wf.amplitude) ** 2
        return probs

    def _generated_distribution(self, params: np.ndarray) -> np.ndarray:
        """Map Born probabilities to the histogram bin space."""
        born = self._generator_circuit(params)
        # If dim matches n_bins, use directly; otherwise interpolate
        if len(born) == self.n_bins:
            return born
        # Simple linear interpolation to n_bins
        x_old = np.linspace(0, 1, len(born))
        x_new = np.linspace(0, 1, self.n_bins)
        dist = np.interp(x_new, x_old, born)
        dist = np.clip(dist, 0, None)
        return dist / (dist.sum() + 1e-12)

    # ------------------------------------------------------------------
    # Discriminator (classical MLP)
    # ------------------------------------------------------------------

    def _init_discriminator(self):
        """Initialize discriminator weights."""
        rng = np.random.default_rng(123)
        h = self.n_bins
        self._disc_W1 = rng.normal(0, 0.3, (self.n_bins, h))
        self._disc_b1 = np.zeros(h)
        self._disc_W2 = rng.normal(0, 0.3, (h, 1))
        self._disc_b2 = np.zeros(1)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    def _discriminator(self, x: np.ndarray) -> float:
        """Discriminator forward pass. Returns P(real)."""
        h = self._relu(x @ self._disc_W1 + self._disc_b1)
        return float(self._sigmoid(h @ self._disc_W2 + self._disc_b2))

    def _train_discriminator_step(self, real_dist: np.ndarray, fake_dist: np.ndarray, lr: float = 0.01):
        """One gradient step for the discriminator (binary cross-entropy)."""
        # Forward
        d_real = self._discriminator(real_dist)
        d_fake = self._discriminator(fake_dist)

        # Numerical gradients for the small MLP
        eps = 1e-5
        params_list = [
            ('W1', self._disc_W1),
            ('b1', self._disc_b1),
            ('W2', self._disc_W2),
            ('b2', self._disc_b2),
        ]

        def loss_fn():
            dr = self._discriminator(real_dist)
            df = self._discriminator(fake_dist)
            # Maximize log(D(real)) + log(1 - D(fake))
            return -(np.log(np.clip(dr, 1e-12, 1)) + np.log(np.clip(1 - df, 1e-12, 1)))

        for name, param in params_list:
            flat = param.ravel()
            grad = np.zeros_like(flat)
            for i in range(len(flat)):
                old = flat[i]
                flat[i] = old + eps
                lp = loss_fn()
                flat[i] = old - eps
                lm = loss_fn()
                flat[i] = old
                grad[i] = (lp - lm) / (2 * eps)
            # Gradient descent (minimize negative → maximize)
            flat -= lr * grad

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _discretize_data(self, data: np.ndarray) -> np.ndarray:
        """Convert continuous data to a normalized histogram."""
        counts, self._bin_edges = np.histogram(data, bins=self.n_bins, density=True)
        dist = counts / (counts.sum() + 1e-12)
        return dist

    def train(
        self,
        real_data: np.ndarray,
        steps: int = 100,
        gen_lr: float = 0.05,
        disc_lr: float = 0.01,
        disc_steps: int = 3,
        callback=None,
    ) -> tuple[list[float], list[float]]:
        """Train the QGAN.

        Args:
            real_data:  1-D array of real data samples.
            steps:      Number of training iterations.
            gen_lr:     Generator learning rate.
            disc_lr:    Discriminator learning rate.
            disc_steps: Discriminator updates per generator update.
            callback:   Optional ``callable(step, gen_loss, disc_loss)``.

        Returns:
            ``(generator_loss_history, discriminator_loss_history)``
        """
        real_data = np.asarray(real_data, dtype=float)
        real_dist = self._discretize_data(real_data)

        # Init
        if self.gen_params is None:
            rng = np.random.default_rng(42)
            self.gen_params = rng.uniform(-np.pi, np.pi, self.n_gen_params)
        self._init_discriminator()

        gen_opt = Adam(lr=gen_lr)
        g_history: list[float] = []
        d_history: list[float] = []

        for step in range(steps):
            fake_dist = self._generated_distribution(self.gen_params)

            # Train discriminator
            for _ in range(disc_steps):
                self._train_discriminator_step(real_dist, fake_dist, disc_lr)

            # Generator loss: -log(D(G(θ)))
            d_fake = self._discriminator(fake_dist)
            g_loss = -np.log(np.clip(d_fake, 1e-12, 1.0))

            d_real = self._discriminator(real_dist)
            d_loss = -(np.log(np.clip(d_real, 1e-12, 1)) + np.log(np.clip(1 - d_fake, 1e-12, 1)))

            g_history.append(float(g_loss))
            d_history.append(float(d_loss))

            # Generator gradient via parameter-shift
            def gen_cost(p):
                fd = self._generated_distribution(p)
                return -np.log(np.clip(self._discriminator(fd), 1e-12, 1.0))

            grad = parameter_shift(gen_cost, self.gen_params)
            self.gen_params = gen_opt.step(self.gen_params, grad)

            if callback is not None:
                callback(step, g_loss, d_loss)

        return g_history, d_history

    def generated_distribution(self, params: np.ndarray | None = None) -> np.ndarray:
        """Return the current generated probability distribution."""
        if params is None:
            params = self.gen_params
        return self._generated_distribution(params)

    def sample(self, n_samples: int, params: np.ndarray | None = None) -> np.ndarray:
        """Draw samples from the generated distribution.

        Args:
            n_samples: Number of samples to draw.
            params:    Generator parameters. Uses trained params if ``None``.

        Returns:
            1-D array of generated samples.
        """
        dist = self.generated_distribution(params)
        if not hasattr(self, '_bin_edges') or self._bin_edges is None:
            # Default bins if no training data was provided
            bin_centers = np.linspace(0, 1, self.n_bins)
        else:
            bin_centers = 0.5 * (self._bin_edges[:-1] + self._bin_edges[1:])
        indices = np.random.choice(len(bin_centers), size=n_samples, p=dist / dist.sum())
        # Add small noise within each bin
        if hasattr(self, '_bin_edges') and self._bin_edges is not None:
            bin_width = self._bin_edges[1] - self._bin_edges[0]
        else:
            bin_width = 1.0 / self.n_bins
        noise = np.random.uniform(-bin_width / 2, bin_width / 2, n_samples)
        return bin_centers[indices] + noise

    def kl_divergence(self, real_data: np.ndarray, params: np.ndarray | None = None) -> float:
        """KL divergence between real and generated distributions.

        Args:
            real_data: 1-D array of real data.

        Returns:
            KL(real || generated).
        """
        real_dist = self._discretize_data(real_data)
        gen_dist = self.generated_distribution(params)
        # Clip to avoid log(0)
        real_dist = np.clip(real_dist, 1e-12, None)
        gen_dist = np.clip(gen_dist, 1e-12, None)
        return float(np.sum(real_dist * np.log(real_dist / gen_dist)))
