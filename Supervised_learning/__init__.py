carbon_source = "glc" # glucose condition
output_name   = "glc" # python v. 3.7.0
import shap # v. 0.41.0
import sklearn # v. 1.0.2
import pandas as pd # v. 1.1.5
import numpy as np # v. 1.21.6
import tensorflow as tf # v. 2.7.1
import keras_tuner as kt # v. 1.1.3
#from h2o4gpu.solvers.elastic_net import ElasticNet # v. 0.20.0
#import h2o4gpu.util.import_data as io
#import h2o4gpu.util.metrics as metrics
import warnings
import random
warnings.filterwarnings(action='ignore')

X_data  = pd.read_feather("input_data/X_train("+carbon_source+").feather").set_index("index")
y_data  = pd.read_feather("input_data/y_train("+carbon_source+").feather").set_index("index")


random_seed       = 0 # fix the seed for reproducability
hp_dir            = "hp_folder" #hyperparameter tuning directory
neurons           = [5,10, 25, 50, 100, 200, 1000,2000] # number of perceptrons for each layers
optimizer_param   = ['adam', 'rmsprop', 'sgd'] # backpropagation optimizers
learning_rate     = [0.1,0.05,0.01,0.005,0.001,0.0001]
kernel_constraint = [-1,2,3,4] # layer weight constraints, -1 : no constraint
dropout           = [0.3,0.4, 0.5, 0.6] # Dropout layer rate
max_trials        = 10000

X_train_scaled = sklearn.preprocessing.StandardScaler().fit_transform(X_data)
y_train = y_data
X_train_scaled, y_train = sklearn.utils.shuffle(X_train_scaled, y_train, random_state=random_seed)


def kernel_constraint_func(int):
    if int == -1:
        return None
    elif int == 2:
        return tf.keras.constraints.max_norm(2)
    elif int == 3:
        return tf.keras.constraints.max_norm(3)
    elif int == 4:
        return tf.keras.constraints.max_norm(4)


def build_model(hp):
    # model construction
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(len(X_data.columns),)))
    for i in range(hp.Int('layers', 0, 4)):
        model.add(tf.keras.layers.Dense(units=hp.Choice('units', neurons), activation='relu',
                                        kernel_constraint=kernel_constraint_func(
                                            hp.Choice("kernel", kernel_constraint))))
        model.add(tf.keras.layers.Dropout(hp.Choice('d_units', dropout)))
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    # optimizer
    optimizer = hp.Choice('optimizer', values=optimizer_param)

    if optimizer == "adam":
        final_optimizer = tf.optimizers.Adam(hp.Choice('learning_rate', values=learning_rate))
    elif optimizer == "sgd":
        final_optimizer = tf.optimizers.SGD(hp.Choice('learning_rate', values=learning_rate))
    elif optimizer == "rmsprop":
        final_optimizer = tf.optimizers.RMSprop(hp.Choice('learning_rate', values=learning_rate))

    # Tune the learning rate for the optimizer
    model.compile(
        optimizer=final_optimizer,
        loss='mse',
        metrics=['mse']
    )
    return model


# Tuning
tuner = kt.RandomSearch(build_model, objective='val_mse',
                        overwrite=True,
                        max_trials=max_trials,
                        executions_per_trial=3,
                        directory=hp_dir,
                        seed=random_seed)

tuner.search(X_train_scaled, y_train, epochs=40, validation_split=0.1, verbose=0)
# tuner.search_space_summary()


# Get the optimal hyperparameters
best_hp = tuner.get_best_hyperparameters()
print("chosen hp:", best_hp[0].values)

