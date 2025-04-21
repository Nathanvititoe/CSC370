import optuna # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from optuna.pruners import MedianPruner # type: ignore
from skorch import NeuralNetClassifier  # type: ignore 
from skorch.callbacks import EpochScoring, PrintLog # type: ignore
from skorch.dataset import ValidSplit # type: ignore
from torch._dynamo import OptimizedModule  # type: ignore
from skorch.callbacks import EarlyStopping # type: ignore

from model_builder import build_model
from dataset_loader import load_dataset


# ignore unnecessary warnings
import warnings
import logging
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Paths and params
full_dataset_path = "./aircraft_dataset/crop"
small_subset = "./aircraft_dataset/small_subset"
data_path = small_subset

# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
img_size = (224, 224)

# ensure it uses gpu acceleration, error otherwise
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise RuntimeError("🚨 CUDA is required but not available!")

# Run an optuna study to get best hyperparams
def get_hyperparams(trial):
    dropout = trial.suggest_float("dropout", 0.25, 0.4) # search for best performing dropout
    lr = trial.suggest_float("lr", 7e-4, 1.2e-3, log=True) # search for best performing LR
    dense_units = trial.suggest_categorical("dense_units", [160, 192, 224]) # search for best performing dense_units, stepping by 64
    batch_size = trial.suggest_categorical("batch_size", [128, 256])

    # load prepared datasets
    dataset, class_names = load_dataset(data_path, img_size)

    # get number of class names
    num_classes = len(class_names)

    # build the model with sample hyperparams
    model = build_model(num_classes=num_classes, dropout_rate=dropout, dense_units=dense_units)
    
    # remove some python overhead to speed up trials/epochs after the first few 
    # longer first trial, speeds up later epochs/trials
    model: OptimizedModule = torch.compile(model, mode="reduce-overhead", disable=True) 

    # Wrap model with Skorch for training/validation
    net = NeuralNetClassifier(
        module=model,  # the built cnn
        criterion=nn.CrossEntropyLoss, # get loss
        optimizer=torch.optim.AdamW,    # use adamW to optimize
        max_epochs=7, # total epochs
        lr=lr, # sample learning rate from Optuna
        device=device, # only train using GPU
        train_split=ValidSplit(0.2, stratified=True, random_state=42), # split the ds
        batch_size=batch_size,
        # speed up dataset loading
        iterator_train__num_workers=8,
        iterator_train__pin_memory=True,
        iterator_valid__num_workers=8,
        iterator_valid__pin_memory=True,
        # logging settings
        verbose=0 , 
        callbacks=[
        # quit early if no val_acc improvement
        EarlyStopping(monitor='valid_loss', patience=3),  # stops if no improvement in 3 epochs
        # log training accuracy
        EpochScoring(scoring='accuracy', lower_is_better=False, name='train_acc', on_train=True),
        
        # log validation accuracy
        EpochScoring(scoring='accuracy', lower_is_better=False, name='val_acc', on_train=False),
        
        # log all values 
        PrintLog(keys_ignored=['batches'])
    ]
    )

    # train the model
    y_labels = torch.tensor(dataset.targets)
    net.fit(dataset, y=y_labels) # no manual splitting, skorch does it

    # evaluate val_loss
    val_loss = max(net.history[:, 'valid_loss'])  # get best/lowest val_loss this trial
    trial.report(val_loss , step=0) # output val loss

    # log the trial results
    print(f"Trial {trial.number}: dropout={dropout:.3f}, lr={lr:.5f}, units={dense_units}, val_loss={val_loss:.4f}")

    # prune poorly performing trials (end early)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_loss  # lower is better

# Run the Optuna study
pruner = MedianPruner(n_warmup_steps=3) # wait 3 steps before pruning
study = optuna.create_study(direction="minimize", pruner=pruner) # minimize val loss in study 
study.optimize(get_hyperparams, n_trials=25) # run 25 trials

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
