import tensorflow as tf
from functools import partial
import os


def convert(model, rep_data, model_save_path):
    def _representative_data_gen(rep_data):
        for input_value, _ in rep_data.take(100):
            yield [input_value]

    _representative_data_gen = partial(_representative_data_gen, rep_data)
    # create the converter object
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # set the optimisations (this means that we can in theory tell the model to optimise for size latency or sparsity but optimisation for size and latency are deprecated while optimisation for sparsity is experimental. In the documentation the suggested option is DEFAULT)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    # set the representative dataset for converter (needed for full-integer quantization)
    converter.representative_dataset = _representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Get the quantized model
    tflite_model_quant = converter.convert()


    # Save the quantized model:
    tflite_model_file = os.path.join(model_save_path, "quant_model.tflite")
    with open(tflite_model_file, 'wb') as f:
        f.write(tflite_model_quant)

    return tflite_model_quant