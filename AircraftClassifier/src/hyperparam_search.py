import optuna # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from optuna.pruners import MedianPruner # type: ignore
from skorch import NeuralNetClassifier  # type: ignore 
from skorch.callbacks import EpochScoring # type: ignore
from skorch.dataset import ValidSplit # type: ignore
from torch._dynamo import OptimizedModule  # type: ignore
from skorch.callbacks import EarlyStopping # type: ignore
import gc

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
    dropout = trial.suggest_float("dropout", 0.345, 0.355) # search for best performing dropout
    lr = trial.suggest_float("lr",4.5e-4, 5.5e-4, log=True) # search for best performing LR
    dense_units = trial.suggest_int("dense_units", 240, 270, step=3) # search for best performing dense_units, stepping by 64
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-5, log=True)
    batch_size = 64

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
        optimizer__weight_decay=weight_decay, # use decay to fight overfitting
        max_epochs=20, # total epochs
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
        verbose=1 , 
        callbacks=[
        # quit early if no val_acc improvement
        EarlyStopping(monitor='valid_acc', patience=5),  # stops if no improvement in 3 epochs
        # log training accuracy
        EpochScoring(scoring='accuracy', lower_is_better=False, name='train_acc', on_train=True),
    ]
    )

    # train the model
    y_labels = torch.tensor(dataset.targets)
    net.fit(dataset, y=y_labels) # no manual splitting, skorch does it


    # evaluate val_acc
    val_acc = max(net.history[:, 'valid_acc'])  # get best/lowest val_acc this trial
    trial.report(val_acc , step=0) # output val_acc

    # log the trial results
    print(f"Trial {trial.number}: dropout={dropout:.3f}, lr={lr:.5f}, units={dense_units}, val_loss={val_acc:.4f}")

    # prune poorly performing trials (end early)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    # clear gpu cache to save vram
    del model, net, dataset, y_labels
    gc.collect()
    torch.cuda.empty_cache()
    

    # return val_acc  
    return val_acc 


# Run the Optuna study
pruner = MedianPruner(n_warmup_steps=3) # wait 3 steps before pruning
study = optuna.create_study(direction="maximize", pruner=pruner) # minimize val loss in study 
study.optimize(get_hyperparams, n_trials=10) # run 10 trials

# Output best hyperparams  found by optuna study
print("Best hyperparameters:")
print(study.best_params)
