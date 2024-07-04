import numpy as np
import tensorflow as tf
import wandb
import pandas as pd
import os
import matplotlib.pyplot as plt

from helper.training_testing import metrics


def test_online(test_ids, y_test, y_pred, online_threshold, stride_minutes):
    # Create a dictionary to group results by IDs
    result_dict = {}
    true_dict = {}

    # Iterate through the IDs and results, and group them together
    for t_id, y_p, y_t in zip(test_ids.numpy(), y_pred, y_test):
        if t_id not in result_dict:
            result_dict[t_id] = []
            true_dict[t_id] = []
        
        result_dict[t_id].append(y_p)
        true_dict[t_id].append(y_t)

    test_id_pos = []
    y_pred_cumulative = []
    y_test_cumulative = []
    y_pred_time = []
    
    kernel = np.ones(online_threshold)
    for k in result_dict.keys():
        pred_array = np.array(result_dict[k])
        conv_result = np.convolve(pred_array.round(), kernel)
        if max(conv_result) >= online_threshold:
            y_pred_cumulative.append(1)
            if true_dict[k][0] == 1:
                # In case true positive time is computed
                sepsis_time = np.where(conv_result >= online_threshold)[0][0]
                y_pred_time.append((pred_array.shape[0] - sepsis_time -1) * stride_minutes / 60) 
            test_id_pos.append(k)
        else:
            y_pred_cumulative.append(0)
        y_test_cumulative.append(true_dict[k][0] == 1)

    return np.array(y_test_cumulative), np.array(y_pred_cumulative), np.array(y_pred_time), np.array(test_id_pos)


def test_float_model(model, test_data, save_folder, online=False, test_ids=None, online_threshold=4, sample_per_hour=30):
    # model.evaluate(test_data)
    y_pred = model.predict(test_data).flatten()
    y_test = np.concatenate([label for _, label in test_data]).flatten()
    # y_pred_digit = y_pred.round()

    pd.DataFrame({"test_ids": tf.squeeze(test_ids), "y_test": y_test, "y_pred": y_pred}).to_csv(os.path.join(save_folder, "results_float.csv"))

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss = loss(y_test, y_pred).numpy()

    wandb_log_dict = {"float/loss": loss}

    if online:
        y_test, y_pred, sepsis_time, _ = test_online(test_ids, y_test, y_pred, online_threshold, sample_per_hour)
        wandb_log_dict["float/sepsis_time_median"] = np.median(sepsis_time)
        wandb_log_dict["float/sepsis_time_std"] = np.std(sepsis_time)

    accuracy, _, _ = metrics.accuracy(y_test, y_pred)

    wandb_log_dict["float/accuracy"] = accuracy

    _, _, auroc, _, roc = metrics.roc_curves(y_test, y_pred)
    _, _, auprc, _, pr = metrics.pr_curves(y_test, y_pred)

    wandb_log_dict["float/AUROC"] = auroc
    wandb_log_dict["float/AUPRC"] = auprc

    roc.plot()
    plt.savefig(os.path.join(save_folder, "roc_float.png"))
    pr.plot()
    plt.savefig(os.path.join(save_folder, "pr_float.png"))

    wandb.log({"float/roc": wandb.Image(os.path.join(save_folder, "roc_float.png"))})
    wandb.log({"float/pr": wandb.Image(os.path.join(save_folder, "pr_float.png"))})

    confusion_matrix = metrics.conf_matrix(y_test, y_pred)

    wandb_log_dict["float/TP"] = confusion_matrix[1,1]
    wandb_log_dict["float/FP"] = confusion_matrix[0,1]
    wandb_log_dict["float/TN"] = confusion_matrix[0,0]
    wandb_log_dict["float/FN"] = confusion_matrix[1,0]

    sensitivity = metrics.sensitivity(confusion_matrix[1,1], confusion_matrix[1,0])
    specificity = metrics.specificity(confusion_matrix[0,0], confusion_matrix[0,1])

    wandb_log_dict["float/sensitivity"] = sensitivity
    wandb_log_dict["float/specificity"] = specificity

    wandb.log(wandb_log_dict)

    return accuracy, loss, auroc, auprc, confusion_matrix


# Helper function to run inference on a TFLite model
def run_tflite_model(quantized_model, test_data):

    # Initialize the interpreter for the converted TFLite model
    interpreter = tf.lite.Interpreter(model_content=quantized_model)
    # Allocate memory for the input and output tensors
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    output = []

    for test_data, _ in test_data:
        for test_tensor in test_data:
            # Rescale the input to int8
            input_scale, input_zero_point, _ = input_details["quantization_parameters"].values() # (https://www.tensorflow.org/lite/api_docs/python/tf/lite/Interpreter#get_input_details) Quantization parameters returns the parameters necessary to convert the input tensor from float32 to int8. It returns scale, zero point and quantized dimension
            test_tensor = (test_tensor / input_scale) + input_zero_point # equation to convert from float32 to int8  (https://www.tensorflow.org/lite/performance/quantization_spec)

            # adjust dimension of input tensor
            test_tensor = np.expand_dims(test_tensor, axis=0).astype(input_details["dtype"])
            # give input to interpreter
            interpreter.set_tensor(input_details["index"], test_tensor)
            # run inference
            interpreter.invoke()
            # get result of inference
            output.append(interpreter.get_tensor(output_details["index"])[0])

    return np.array(output)


# Helper function to evaluate a TFLite model on all images
def evaluate_quantized_model(quantized_model, test_data, save_folder, online=False, test_ids=None, online_threshold=4, stride_minutes=30):
    y_test = np.concatenate([label for _, label in test_data]).flatten()

    #Get predictions
    y_pred = run_tflite_model(quantized_model, test_data).flatten()
    
    pd.DataFrame({"y_test": tf.squeeze(y_test), "y_pred": y_pred}).to_csv(os.path.join(save_folder, "results_quant.csv"))

    y_pred = (y_pred+128)/255
    # y_pred_digit = y_pred.round().astype(int)

    wandb_log_dict = {}

    if online:
        y_test, y_pred, sepsis_time, _ = test_online(test_ids, y_test, y_pred, online_threshold, stride_minutes)
        wandb_log_dict["quant/sepsis_time_median"] = np.median(sepsis_time)
        wandb_log_dict["quant/sepsis_time_std"] = np.std(sepsis_time)

    # Get accuracy [%]
    accuracy, _, _ = metrics.accuracy(y_test, y_pred)
    wandb_log_dict["quant/accuracy"] = accuracy

    # Get AUROC and AUPRC
    _, _, auroc, _, roc = metrics.roc_curves(y_test, y_pred)
    _, _, auprc, _, pr = metrics.pr_curves(y_test, y_pred)
    wandb_log_dict["quant/AUROC"] = auroc
    wandb_log_dict["quant/AUPRC"] = auprc

    roc.plot()
    plt.savefig(os.path.join(save_folder, "roc_quant.png"))
    pr.plot()
    plt.savefig(os.path.join(save_folder, "pr_quant.png"))

    wandb.log({"float/roc": wandb.Image(os.path.join(save_folder, "roc_quant.png"))})
    wandb.log({"float/pr": wandb.Image(os.path.join(save_folder, "pr_quant.png"))})

    # Get confusion matrix
    confusion_matrix = metrics.conf_matrix(y_test, y_pred)
    wandb_log_dict["quant/TP"] = confusion_matrix[1,1]
    wandb_log_dict["quant/FP"] = confusion_matrix[0,1]
    wandb_log_dict["quant/TN"] = confusion_matrix[0,0]
    wandb_log_dict["quant/FN"] = confusion_matrix[1,0]

    sensitivity = metrics.sensitivity(confusion_matrix[1,1], confusion_matrix[1,0])
    specificity = metrics.specificity(confusion_matrix[0,0], confusion_matrix[0,1])

    wandb_log_dict["quant/sensitivity"] = sensitivity
    wandb_log_dict["quant/specificity"] = specificity

    # Log metrics on wandb
    wandb.log(wandb_log_dict)