# test_tf_keras.py
try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    from tensorflow import keras # type: ignore
    print("Keras version:", keras.__version__)
except Exception as e:
    print("Error:", e)