#testing gpu

import tensorflow as tf

# Check if TensorFlow can see the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check if TensorFlow is using GPU
if tf.test.is_built_with_cuda():
    print("TensorFlow is built with CUDA")
else:
    print("TensorFlow is not built with CUDA")

if tf.test.is_gpu_available():
    print("GPU is available")
else:
    print("GPU is not available")

# Try to perform a simple operation on GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print("Matrix multiplication result:")
    print(c)

# Print device placement
print("\nDevice placement:")
tf.debugging.set_log_device_placement(True)
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)
