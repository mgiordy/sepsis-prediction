import numpy as np
import tensorflow as tf
from wandb.keras import WandbMetricsLogger
from functools import partial

def learning_rate_schedule(epoch, lr, step_lr_epoch_div, step_lr_div_factor):
    if epoch%step_lr_epoch_div == 0 and epoch != 0:
        return lr * step_lr_div_factor
    else:
        return lr


def train_model(model, train_ds, epoches, lr, step_lr_epoch_div, step_lr_div_factor, weight_decay):
    #create loss function and optimizer
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay, epsilon=1e-8)

    lr_schedule = partial(learning_rate_schedule, step_lr_epoch_div=step_lr_epoch_div, step_lr_div_factor=step_lr_div_factor)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='ROC-AUC'), tf.keras.metrics.AUC(curve='PR', name='PR-AUC')])

    model.fit(train_ds, epochs=epoches, callbacks=[WandbMetricsLogger('epoch'), lr_scheduler])

    return model