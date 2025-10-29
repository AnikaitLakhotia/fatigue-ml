# src/eeg/models/ssl_tf.py
from __future__ import annotations
"""
Contrastive SSL (SimCLR / NT-Xent) implemented in TensorFlow/Keras.

Components:
 - TemporalConvEncoderTF: Keras encoder mapping (B, C, T) -> (B, enc_dim)
 - ProjectionHeadTF: small MLP projecting encoder outputs
 - NTXentLossTF: vectorized contrastive loss
 - ContrastiveTrainerTF: training manager with checkpoints and metrics
"""
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import time
import tensorflow as tf
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalConvEncoderTF(tf.keras.Model):
    """
    Lightweight temporal conv encoder for EEG epochs.
    Input: (batch, channels, samples)
    Output: embedding (batch, enc_dim)
    """

    def __init__(self, n_channels: int, n_samples: int, enc_dim: int = 128, hidden_filters: int = 64, kernel_size: int = 9):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(hidden_filters, kernel_size, padding="same", activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(hidden_filters, kernel_size, padding="same", activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(enc_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(enc_dim)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = tf.transpose(x, perm=[0, 2, 1])  # (B,C,T) -> (B,T,C)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class ProjectionHeadTF(tf.keras.Model):
    """MLP projection head mapping encoder embeddings -> proj_dim"""

    def __init__(self, in_dim: int, proj_dim: int = 64):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(max(in_dim, 128), activation="relu")
        self.fc2 = tf.keras.layers.Dense(proj_dim)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class NTXentLossTF(tf.keras.losses.Loss):
    """NT-Xent loss for contrastive learning."""

    def __init__(self, temperature: float = 0.1, name: str | None = None):
        super().__init__(name=name)
        self.temperature = temperature
        self.eps = 1e-8

    def call(self, z1: tf.Tensor, z2: tf.Tensor) -> tf.Tensor:
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)
        batch_size = tf.shape(z1)[0]
        z = tf.concat([z1, z2], axis=0)  # (2B, D)
        sim = tf.matmul(z, z, transpose_b=True) / self.temperature
        sim = sim - tf.reduce_max(sim, axis=1, keepdims=True)
        mask = 1.0 - tf.eye(2 * batch_size)
        exp_sim = tf.exp(sim) * mask
        pos = tf.concat([
            tf.linalg.diag_part(sim[:batch_size, batch_size:]),
            tf.linalg.diag_part(sim[batch_size:, :batch_size])
        ], axis=0)
        numerator = tf.exp(pos)
        denominator = tf.reduce_sum(exp_sim, axis=1) + self.eps
        loss = -tf.math.log(numerator / denominator)
        return tf.reduce_mean(loss)


class ContrastiveTrainerTF:
    """Trainer for contrastive SSL using TensorFlow."""

    def __init__(
        self,
        encoder: TemporalConvEncoderTF,
        projector: ProjectionHeadTF,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        temperature: float = 0.1,
        save_dir: Optional[Path] = None,
    ):
        self.encoder = encoder
        self.projector = projector
        self.loss_fn = NTXentLossTF(temperature=temperature)
        self.lr = learning_rate
        self.wd = weight_decay
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.save_dir = Path(save_dir) if save_dir else None
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder, projector=self.projector)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, str(self.save_dir), max_to_keep=5) if self.save_dir else None

    @tf.function
    def _train_step(self, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            z1_enc = self.encoder(x1, training=True)
            z2_enc = self.encoder(x2, training=True)
            p1 = self.projector(z1_enc, training=True)
            p2 = self.projector(z2_enc, training=True)
            loss = self.loss_fn(p1, p2)
        variables = self.encoder.trainable_variables + self.projector.trainable_variables
        grads = tape.gradient(loss, variables)
        if self.wd > 0:
            grads = [g + self.wd * v if g is not None else None for g, v in zip(grads, variables)]
        self.optimizer.apply_gradients(zip(grads, variables))
        self.train_loss.update_state(loss)
        return loss

    def train(
        self,
        dataset: tf.data.Dataset,
        epochs: int = 10,
        steps_per_epoch: Optional[int] = None,
        save_every_n_epochs: int = 1,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        if steps_per_epoch is None:
            try:
                steps_per_epoch = int(tf.data.experimental.cardinality(dataset).numpy())
            except Exception:
                steps_per_epoch = None

        history = {"loss": []}
        total_steps = 0
        for ep in range(1, epochs + 1):
            self.train_loss.reset_state()
            start = time.time()
            step = 0
            for x1, x2 in dataset:
                loss = self._train_step(x1, x2)
                step += 1
                total_steps += 1
                if steps_per_epoch and step >= steps_per_epoch:
                    break
                if verbose and step % 50 == 0:
                    logger.info("Epoch %d step %d loss=%.6f", ep, step, float(self.train_loss.result()))
            epoch_loss = float(self.train_loss.result())
            history["loss"].append(epoch_loss)
            logger.info("Epoch %d/%d done: loss=%.6f (%.1fs)", ep, epochs, epoch_loss, time.time() - start)
            if self.ckpt_manager and ep % save_every_n_epochs == 0:
                ckpt_path = self.ckpt_manager.save()
                logger.info("Saved checkpoint: %s", ckpt_path)

        if self.save_dir:
            enc_path = Path(self.save_dir) / "encoder.weights.h5"
            self.encoder.save_weights(str(enc_path))
            proj_path = Path(self.save_dir) / "projector.weights.h5"
            self.projector.save_weights(str(proj_path))
            logger.info("Saved final encoder/projector weights to %s", str(self.save_dir.resolve()))

        return {"history": history, "steps": total_steps, "checkpoint_dir": str(self.save_dir.resolve()) if self.save_dir else None}