# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:47:24 2023

@author: Mert
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss

np.random.seed(11)

data, features, dtypes, target_scale, task = load_dataset(
    dataset=args.dataset
    )

X, y = data[0][0], data[0][1]
data = pd.DataFrame(np.concatenate([X, y.reshape(-1,1)], -1))
data = data.rename(columns={0:'x1', 1:'x2', 2:'x3', 3:'y'})
data.head()

cat_cols = []

num_cols = ['x1', 'x2', 'x3']
target=["y"]

train, test = train_test_split(
    data, test_size=0.2, random_state=11
    )

from lightgbm import LGBMRegressor
from sklearn.preprocessing import OrdinalEncoder

# LightGBM needs categorical columns encoded as integers
train_enc = train.copy()
test_enc = test.copy()
for col in cat_cols:
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", 
        encoded_missing_value=np.nan, 
        unknown_value=np.nan
        )
    train_enc[col] = enc.fit_transform(train_enc[col].values.reshape(-1,1))
    test_enc[col] = enc.transform(test_enc[col].values.reshape(-1,1))
    
clf = LGBMRegressor(random_state=42)
clf.fit(train_enc.drop(columns=target[0]), train_enc[target], categorical_feature=cat_cols)
test_pred = clf.predict(test_enc.drop(columns=target[0]))
test_pred_proba = clf.predict(test_enc.drop(columns=target[0]))

acc = mean_squared_error(test[target[0]].values, test_pred) ** 0.5
# loss = log_loss(test[target[0]].values, test_pred_proba)
print(f"Acc: {acc}")

from pytorch_tabular import TabularModel
from pytorch_tabular.models import (
    CategoryEmbeddingModelConfig, 
    FTTransformerConfig, 
    TabNetModelConfig, 
    GatedAdditiveTreeEnsembleConfig, 
    TabTransformerConfig, 
    AutoIntConfig
)
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig

data_config = DataConfig(
    target=target, #target should always be a list.
    continuous_cols=num_cols,
    categorical_cols=cat_cols,
)

trainer_config = TrainerConfig(
#     auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=10000,
    early_stopping="valid_loss", # Monitor valid_loss for early stopping
    early_stopping_mode = "min", # Set the mode as min because for val_loss, lower is better
    early_stopping_patience=50, # No. of epochs of degradation training will wait before terminating
    checkpoints="valid_loss", # Save best checkpoint monitoring val_loss
    load_best=True, # After training, load the best checkpoint
)

optimizer_config = OptimizerConfig()

head_config = LinearHeadConfig(
    layers="", # No additional layer in head, just a mapping layer to output_dim
    dropout=0.1,
    initialization="kaiming"
).__dict__ # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)

model_config = CategoryEmbeddingModelConfig(
    task="regression",
    layers="64-32",  # Number of nodes in each layer
    activation="ReLU", # Activation between each layers
    learning_rate = 1e-3,
    head = "LinearHead", #Linear Head
    head_config = head_config, # Linear Head Config
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
# tabular_model.fit(train=train)
# tabular_model.evaluate(test)

model_config = GatedAdditiveTreeEnsembleConfig(
    task="regression",
    learning_rate = 1e-3,
    head = "LinearHead", #Linear Head
    head_config = head_config, # Linear Head Config
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(train=train)
tabular_model.evaluate(test)



