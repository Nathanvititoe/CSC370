This Assignment is using https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data, I'm not tracking it through github because it is 10.2gb of files, it will need to be manually downloaded and unzipped to '/aircraft_dataset'
![image](https://github.com/user-attachments/assets/ff7661c8-1138-4070-9841-6d3ef51d9f74)


Started using Tensorflow CPU/Keras to learn/train, but maxed out around 60% Training accuracy and 50% Validation Accuracy 

Training was taking too long to tune it more or run hyperparameter searches effectively

Switched to Pytorch/Skorch and GPU acceleration (pytorch was much easier to setup, tensorflow consistently gave me versioning issues)

Used Optuna to run hyperparameter search and utilized Pretrained models as a base for finding important features and a custom top/output layer to determine what those features mean for identification

    With EfficientNetB3, 128x128 resolution, and Optuna maximizing val_acc the best Optuna Trial returned a 90% Validation Accuracy

    With EfficientNetB0, 224x224 resolution and Optuna set to minimize val_loss, 

