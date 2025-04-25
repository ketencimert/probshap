# -*- coding: utf-8 -*-
'''
Created on Tue Dec 28 13:52:24 2021

@author: Mert
'''

from argparse import Namespace

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from ray import train, tune, init
from ray.tune import CLIReporter
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from ray.air.integrations.keras import ReportCheckpointCallback

def build_model(
        d_in, layer_size, activation, batchnorm, layernorm,
        dropout_input, dropout_intermediate, task
        ):
    inputs = keras.Input(
        shape=(d_in,),
        name="features"
        )

    if batchnorm:
        inputs =  keras.layers.BatchNormalization()(
            inputs,
            )

    if dropout_input>0:
        inputs = layers.Dropout(
                    dropout_input,
                    name="dropout_input"
                    )(inputs)

    x_ = layers.Dense(
        layer_size[0],
        activation=activation,
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.l2(1e-4),
        activity_regularizer=regularizers.l2(1e-5),
        name="dense_input"
        )(inputs)

    i = 2
    for size in layer_size[1:]:
        if dropout_intermediate>0:
            x_ = layers.Dropout(
                dropout_intermediate,
                input_shape=(size,),
                name="dropout_intermediate_{}".format(i),
                )(x_)

        x_ = layers.Dense(
            size,
            activation=activation,
            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.l2(1e-4),
            activity_regularizer=regularizers.l2(1e-5),
            name="dense_{}".format(i)
            )(x_)

        i += 1
        if layernorm:
            x_ = keras.layers.LayerNormalization()(x_)

    if task=='regression':
        outputs = layers.Dense(1, name="predictions")(x_)

    else:
        outputs = layers.Dense(
            1,
            activation='sigmoid',
            name="predictions"
            )(x_)

    return keras.Model(inputs=inputs, outputs=outputs)

def train_dnn(config):

    d_in = config['x_tr'].shape[1]

    model = build_model(
        d_in, config['layer_size'], config['activation'],
        config['batchnorm'], config['layernorm'],
        config['dropout_input'], config['dropout_intermediate'],
        config['task']
        )

    if config['task']=='regression':
        loss = keras.losses.MeanSquaredError()
        metric = keras.metrics.RootMeanSquaredError()
        monitor = 'val_root_mean_squared_error'

    else:
        loss = keras.losses.BinaryCrossentropy()
        metric = tf.keras.metrics.AUC()
        monitor = 'auc'

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['lr']),
        loss=loss,
        metrics=[metric],
    )

    model.fit(
        config['x_tr'],
        config['y_tr'],
        batch_size=config['batch_size'],
        epochs=config['epochs'] // 5,
        validation_data=(config['x_val'], config['y_val']),
        callbacks=[
            ReportCheckpointCallback(metrics={"objective": monitor})],
        verbose=0
    )

def tune_dnn(args, tr_dataloader, val_dataloader, task):

    config = merge(args, tr_dataloader, val_dataloader, task)
    if config['task']=='regression':
        mode = 'min'
    else:
        mode = 'max'

    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=4)

    sched = AsyncHyperBandScheduler(
        time_attr='training_iteration',
        max_t=config['epochs'] // 5,
        grace_period=config['epochs'] // 50,
        )

    init(
        num_cpus=0,
        num_gpus=1
        )

    tuner = tune.Tuner(
        tune.with_resources(train_dnn, resources={"cpu": 0, "gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="objective",
            mode=mode,
            scheduler=sched,
            num_samples=100,
            search_alg=algo,
        ),
        run_config=train.RunConfig(
            name="exp",
            stop={"objective": 0.99},
        ),
        param_space=config,
    )
    results = tuner.fit()

    args = results.get_best_result().config
    for key in ['x_tr', 'y_tr', 'x_val', 'y_val', 'input_size', 'task']:
        args.pop(key)

    return Namespace(**args)

def merge(args, tr_dataloader, val_dataloader, task):

    x_tr, y_tr = tr_dataloader.dataset.__getds__()
    x_val, y_val = val_dataloader.dataset.__getds__()

    config = {
    	'batch_size': tune.choice(
            [128, 512, 1024]
            ),
        'lr': tune.choice(
            [1e-4, 5e-4, 1e-3]
            ),
        'dropout_input': tune.choice(
            [0, 1e-1, 3e-1, 5e-1]
            ),
        'dropout_intermediate': tune.choice(
            [0, 1e-1, 3e-1, 5e-1]
            ),
    	'layernorm': tune.choice(
            [True, False]
            ),
    	'batchnorm': tune.choice(
            [True, False]
            ),
        'activation': tune.choice(
            ['elu', 'selu']
            ),
        'layer_size': tune.choice(
            [
                [312],
                [128],
                [32],
                [32, 32],
                [32, 32, 32],
                [100, 100],
                [100, 100, 100],
                [100, 200, 100],
                [64, 128],
                [128, 64],
                ]
            )
    }

    config['tune'] = args.tune
    config['cv_folds'] = args.cv_folds
    config['preprocess'] = args.preprocess
    config['dataset'] = args.dataset
    config['tensor_dtype'] = args.tensor_dtype
    config['epochs'] = args.epochs

    config['input_size'] = x_tr.shape[1]
    config['task'] = task
    config['x_tr'] = x_tr
    config['y_tr'] = y_tr
    config['x_val'] = x_val
    config['y_val'] = y_val

    return config
