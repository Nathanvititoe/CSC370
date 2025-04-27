import tensorflow as tf
import gc

def cleanup():
    # clear TensorFlow session and reset 
    tf.keras.backend.clear_session()
    print("\u2713 -- Tensorflow Session cleared")

    # reset tensorflow-cpu graph
    tf.compat.v1.reset_default_graph()
    print("\u2713 -- Reset TensorFlow default graph.")

    # clear CUDA cache when using gpu
    if tf.config.list_physical_devices('GPU'):
        tf.compat.v1.reset_default_graph()
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    # clean up unused memory
    print("\u2713 -- Garbage Collected")
    gc.collect()
    

