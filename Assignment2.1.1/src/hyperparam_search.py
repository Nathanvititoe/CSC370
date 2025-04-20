import optuna # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from optuna.pruners import MedianPruner # type: ignore
from skorch import NeuralNetClassifier  # type: ignore 
from skorch.callbacks import EpochScoring, PrintLog # type: ignore

from model_builder import build_model
from dataset_loader import load_dataset


# ignore unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Paths and params
full_dataset_path = "./aircraft_dataset/crop"
subset_path = "./aircraft_dataset/aircraft_subset"
small_subset = "./aircraft_dataset/small_subset"
data_path = full_dataset_path

img_size = (128, 128)

# ensure it uses gpu acceleration, error otherwise
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise RuntimeError("🚨 CUDA is required but not available!")

# Run an optuna study to get best hyperparams
def get_hyperparams(trial):
    dropout = trial.suggest_float("dropout", 0.15, 0.35) # search between 0.15-0.35 dropout
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True) # search between 1e-4-3e-4 for LR
    dense_units = trial.suggest_int("dense_units", 128, 512, step=64) # search between 128-512 dense_units, stepping by 64
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # load prepared datasets
    train_loader, val_loader, class_names = load_dataset(data_path, img_size, batch_size)

    # pass skorch the raw ds only
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset

    # get number of class names
    num_classes = len(class_names)

    # build the model with sample hyperparams
    model = build_model(num_classes=num_classes, dropout_rate=dropout, dense_units=dense_units)

    # Wrap model with Skorch for training/validation
    net = NeuralNetClassifier(
        module=model,  # the built cnn
        criterion=nn.CrossEntropyLoss, # get loss
        optimizer=torch.optim.Adam,    # use adam to optimize
        max_epochs=15, # total epochs
        lr=lr, # sample learning rate from Optuna
        device=device, # only train using GPU
        train_split=None, # disable internal split (we did this in load_dataset)
        batch_size=batch_size,
        verbose=1, 
        callbacks=[
        EpochScoring(scoring='accuracy', lower_is_better=False, name='val_acc', on_train=False),
        PrintLog()
    ]
    )

    # train the model
    net.fit(train_ds, y=None)

    # evaluate val_acc
    val_acc = net.score(val_ds, y=None)  # get acc score using skorch
    trial.report(val_acc , step=0) # output val acc

    # log the trial results
    print(f"Trial {trial.number}: dropout={dropout:.3f}, lr={lr:.5f}, units={dense_units}, val_acc={val_acc:.4f}")

    # prune poorly performing trials (end early)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_acc  # lower is better

# Run the Optuna study
pruner = MedianPruner(n_warmup_steps=3) # wait 3 steps before pruning
study = optuna.create_study(direction="maximize", pruner=pruner) # maximize val acc in study 
study.optimize(get_hyperparams, n_trials=20) # run 20 trials

# Output best hyperparams  found by optuna study
print("Best hyperparameters:")
print(study.best_params)


# Dont use tensorflow anymore, too complicated to setup for gpu acceleration
# import optuna # type: ignore # hyperparam optimizer
# from dataset_loader import load_dataset # dataset loader
# from model_builder import build_pretrained_model # model builder
# from optuna.integration import TFKerasPruningCallback # type: ignore
# from optuna.pruners import MedianPruner  # type: ignore
# import warnings

# # early stopping for over/underfitting and learning rate reducer
# import tensorflow as tf # type: ignore
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
# warnings.filterwarnings("ignore")

# # attempted to learn optuna to figure out the best hyperparameter combination, was unsuccessful as of 20APR
# full_dataset_path = "./aircraft_dataset/crop"
# subset_path = "./aircraft_dataset/aircraft_subset"
# small_subset = "./aircraft_dataset/small_subset"

# data_path = subset_path
# img_size = (224, 224)

# def get_hyperparams(trial):
#     # Try combos of these hyperparameters in these ranges randomly
#     dropout = trial.suggest_float('dropout', 0.15, 0.35) # tests dropping different % of data 
#     lr = trial.suggest_float('lr', 1e-3, 3e-3, log=True) # tests best learning rate
#     units = trial.suggest_int('dense_units', 128, 512, step=64)
#     batch_size = 16

#     # Load dataset with sampled batch size
#     train_ds, val_ds, class_names = load_dataset(data_path, img_size, batch_size)
#     num_classes = len(class_names)

#     # Build model with sampled dropout and learning rate
#     model = build_pretrained_model(input_shape=img_size + (3,), num_classes=num_classes,
#                         dropout_rate=dropout, learning_rate=lr, dense_units=units)

#     # define our early stop and LR reducer
#     early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

#     # train the model  w/ 15 epochs, add pruning to drop poor performing samples earlier
#     model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stop, reduce_lr,TFKerasPruningCallback(trial, 'val_accuracy')], verbose=0)

#     # test validation 
#     val_loss, val_acc = model.evaluate(val_ds, verbose=1)
#     return val_loss # minimize val_loss w/ optuna study 


# pruner = MedianPruner(n_warmup_steps=3)

# study = optuna.create_study(direction="minimize", pruner=pruner) 

# # run the study
# study.optimize(get_hyperparams, n_trials=20)

# print("Best hyperparameters:")
# print(study.best_params) # output best hyperparams from all trials
