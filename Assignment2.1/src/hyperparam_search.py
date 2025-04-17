import optuna # hyperparam optimizer
from dataset_loader import load_dataset # dataset loader
from model_builder import build_pretrained_model # model builder
from optuna.integration import TFKerasPruningCallback
from optuna.pruners import MedianPruner
from dataset_setup import get_generators
import os

# early stopping for over/underfitting and learning rate reducer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import tensorflow as tf

full_dataset_path = "./aircraft_dataset/crop"
subset_path = "./aircraft_dataset/aircraft_subset"
small_subset = "./aircraft_dataset/small_subset"

data_path = small_subset
img_size = (256, 256)

def get_hyperparams(trial):
    # Sample hyperparameters
    dropout = trial.suggest_float('dropout', 0.15, 0.35) # tests dropping different % of data 
    lr = trial.suggest_float('lr', 1e-3, 3e-3, log=True) # tests best learning rate
    # batch_size = trial.suggest_categorical('batch_size', [16]) # adjusts batch size 
    batch_size = 16

    # Load dataset with sampled batch size
    # train_ds, val_ds, class_names = load_dataset(data_path, img_size, batch_size)

    train_ds, val_ds, class_names = get_generators(data_path, img_size, batch_size)
    num_classes = len(class_names)
    # Build model with sampled dropout and learning rate
    model = build_pretrained_model(input_shape=img_size + (3,), num_classes=num_classes,
                        dropout_rate=dropout, learning_rate=lr)

    # define our early stop and LR reducer
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

    # train the model  w/ 20 epochs
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stop, reduce_lr,TFKerasPruningCallback(trial, 'val_accuracy')], verbose=1)

    # test validation 
    val_loss, val_acc = model.evaluate(val_ds, verbose=1)
    return val_loss, val_acc  

if __name__ == "__main__":
    storage = "sqlite:///../optuna_logs/aircraft_optim.db"
    study_name = "aircraft_optimization"
    pruner = MedianPruner(n_warmup_steps=3)

    try:
        # Try loading the study if it exists
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        # If not, create it
        study = optuna.create_study(study_name=study_name, storage=storage, direction='maximize', pruner=pruner)

    # Run the study
    study.optimize(get_hyperparams, n_trials=20)

    print("Best hyperparameters:")
    print(study.best_params) # output best hyperparams
