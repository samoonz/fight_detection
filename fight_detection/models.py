from __future__ import annotations

from typing import Any

from tensorflow import keras
from tensorflow.keras import layers


def _compile(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pure_lstm(timesteps: int, frame_dim: int, cfg: dict[str, Any]) -> keras.Model:
    inp = keras.Input((timesteps, frame_dim), name="skeleton_sequence")
    x = layers.Masking(mask_value=0.0)(inp)
    for index in range(4):
        x = layers.LSTM(int(cfg["rnn_units"]), return_sequences=index < 3)(x)
        x = layers.Dropout(float(cfg["dropout"]))(x)
    out = layers.Dense(2, activation="softmax")(x)
    return _compile(
        keras.Model(inp, out, name="Pure_LSTM"),
        float(cfg["learning_rate"]),
    )


def build_bilstm_lstm(timesteps: int, frame_dim: int, cfg: dict[str, Any]) -> keras.Model:
    inp = keras.Input((timesteps, frame_dim), name="skeleton_sequence")
    x = layers.Masking(mask_value=0.0)(inp)

    x = layers.Bidirectional(
        layers.LSTM(int(cfg["rnn_units"]), return_sequences=True)
    )(x)
    x = layers.Dropout(float(cfg["dropout"]))(x)

    x = layers.Bidirectional(
        layers.LSTM(int(cfg["rnn_units"]), return_sequences=True)
    )(x)
    x = layers.Dropout(float(cfg["dropout"]))(x)

    x = layers.LSTM(int(cfg["rnn_units"]), return_sequences=True)(x)
    x = layers.Dropout(float(cfg["dropout"]))(x)

    x = layers.LSTM(int(cfg["rnn_units"]), return_sequences=False)(x)
    x = layers.Dropout(float(cfg["dropout"]))(x)

    out = layers.Dense(2, activation="softmax")(x)
    return _compile(
        keras.Model(inp, out, name="Fusion_BiLSTM_LSTM"),
        float(cfg["learning_rate"]),
    )


def build_bilstm_gru(timesteps: int, frame_dim: int, cfg: dict[str, Any]) -> keras.Model:
    inp = keras.Input((timesteps, frame_dim), name="skeleton_sequence")
    x = layers.Masking(mask_value=0.0)(inp)

    x = layers.Bidirectional(
        layers.LSTM(int(cfg["rnn_units"]), return_sequences=True)
    )(x)
    x = layers.Dropout(float(cfg["dropout"]))(x)

    x = layers.Bidirectional(
        layers.LSTM(int(cfg["rnn_units"]), return_sequences=True)
    )(x)
    x = layers.Dropout(float(cfg["dropout"]))(x)

    x = layers.GRU(int(cfg["rnn_units"]), return_sequences=True)(x)
    x = layers.Dropout(float(cfg["dropout"]))(x)

    x = layers.GRU(int(cfg["rnn_units"]), return_sequences=False)(x)
    x = layers.Dropout(float(cfg["dropout"]))(x)

    out = layers.Dense(2, activation="softmax")(x)
    return _compile(
        keras.Model(inp, out, name="Fusion_BiLSTM_GRU"),
        float(cfg["learning_rate"]),
    )


MODEL_BUILDERS = {
    "pure_lstm": build_pure_lstm,
    "bilstm_lstm": build_bilstm_lstm,
    "bilstm_gru": build_bilstm_gru,
}
