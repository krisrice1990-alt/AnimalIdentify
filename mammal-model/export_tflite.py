"""Export the trained numpy MLP head to a quantized TFLite model.

Builds an identical Keras model (Dense(hidden, relu) -> Dense(6, softmax);
dropout is training-only and correctly absent at inference), copies the numpy
weights, converts with dynamic-range quantization, and parity-checks the
.tflite against the numpy forward pass with ai_edge_litert.

Requires: tensorflow  (pip install tensorflow-cpu)

Outputs:
    <out>/mammal-head.tflite        quantized head (~tens of KB)
    <out>/export_report.json        sizes + parity check numbers
"""
import argparse
import json
import os
import numpy as np

from config import N_CLASSES


def build_keras(in_dim, hidden, W1, b1, W2, b2):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(in_dim,)),
        tf.keras.layers.Dense(hidden, activation="relu", name="fc1"),
        tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="fc2"),
    ])
    model.get_layer("fc1").set_weights([W1.astype(np.float32), b1.astype(np.float32)])
    model.get_layer("fc2").set_weights([W2.astype(np.float32), b2.astype(np.float32)])
    return model


def tflite_parity(tflite_path, X, P_numpy):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError as exc:
            raise SystemExit("ERROR: parity checking needs ai-edge-litert or tflite-runtime.") from exc
    it = Interpreter(model_path=tflite_path)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    P_tfl = np.empty_like(P_numpy, dtype=np.float32)
    for i in range(len(X)):
        it.set_tensor(inp, X[i:i + 1].astype(np.float32))
        it.invoke()
        P_tfl[i] = it.get_tensor(outp)[0]
    diff = np.abs(P_tfl - P_numpy)
    return {"max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "argmax_agreement": float((P_tfl.argmax(1) == P_numpy.argmax(1)).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cache", required=True, help="feature cache for parity check")
    ap.add_argument("--out", required=True, help="output .tflite path")
    args = ap.parse_args()

    import tensorflow as tf  # noqa: F401  (fail fast with a clear error if missing)
    z = np.load(args.weights, allow_pickle=True)
    in_dim, hidden = int(z["in_dim"]), int(z["hidden"])
    from config import CLASSES
    if "classes" in z and list(z["classes"].astype(str)) != CLASSES:
        raise SystemExit("ERROR: weights class order does not match this pipeline's class contract.")
    for key, shape in (("W1", (in_dim, hidden)), ("b1", (hidden,)),
                       ("W2", (hidden, N_CLASSES)), ("b2", (N_CLASSES,))):
        if key not in z or z[key].shape != shape or not np.isfinite(z[key]).all():
            raise SystemExit(f"ERROR: invalid weights array {key}; expected shape {shape} with finite values.")
    model = build_keras(in_dim, hidden, z["W1"], z["b1"], z["W2"], z["b2"])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]   # dynamic-range quant
    tflite_bytes = converter.convert()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(tflite_bytes)
    print(f"wrote {args.out}  ({len(tflite_bytes) / 1024:.1f} KiB)")

    # parity check on the val split
    c = np.load(args.cache, allow_pickle=True)
    Xva = c["X"][c["split"] == 1].astype(np.float32)
    if not len(Xva):
        raise SystemExit("ERROR: cache has no validation samples; refusing export without parity-check data.")
    if Xva.ndim != 2 or Xva.shape[1] != in_dim or not np.isfinite(Xva).all():
        raise SystemExit("ERROR: validation features do not match trained model input shape or contain NaN/Inf.")
    P_numpy = model.predict(Xva, verbose=0)
    parity = tflite_parity(args.out, Xva, P_numpy)
    print("parity vs numpy/Keras:", parity)
    if parity["argmax_agreement"] < 0.99 or parity["max_abs_diff"] > 0.03:
        raise SystemExit("ERROR: exported TFLite failed parity thresholds; do not deploy this model.")
    with open(os.path.splitext(args.out)[0] + "_export_report.json", "w") as f:
        json.dump({"tflite_bytes": len(tflite_bytes),
                   "in_dim": in_dim, "hidden": hidden, "n_classes": N_CLASSES,
                   "parity": parity}, f, indent=1)


if __name__ == "__main__":
    main()
