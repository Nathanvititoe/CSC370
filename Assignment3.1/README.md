The dataset I used is located here : https://www.kaggle.com/datasets/nathanvititoe/military-aircraft-datasetsubset/data

Used a Convolutional Neural Network to identify aircraft types, and classify the images: 
https://www.ibm.com/think/topics/convolutional-neural-networks
https://www.geeksforgeeks.org/introduction-convolution-neural-network/
https://open.spotify.com/episode/3WloHMXls2B7urlrcL6cCH?si=PmHsO3ThQi2QbKbqWAYG6g

Utilized Tranfer Learning and MobileNetV2 in order to get higher accuracy:
https://www.ibm.com/think/topics/transfer-learning#:~:text=Transfer%20learning%20is%20a%20machine,2
https://keras.io/api/applications/
https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
https://keras.io/guides/transfer_learning/


 Tested Different pre-trained model validation accuracy
    --maybe test mobile net lite for speed/accuracy, only available through tensorflowhub api

     **EfficientNetB0 - Designed for 224x224
        (224x224) : 93.5% accuracy- small_subset/10 epochs
        (96x96)   : 85.4% accuracy- small_subset/10 epochs

     *EfficientNetB3 - Designed for 300x300
        (300x300) : 91.2% accuracy- small_subset/10 epochs
        (224x224) : 92.5% accuracy- small_subset/20 epochs
        (224x224) : 91.3% accuracy- small_subset/10 epochs
        (96x96)   : 86% accuracy- small_subset/10 epochs   
    
    *EfficientNetV2B0 - Designed for 224x224
        (224x224) : 92% accuracy- small_subset/10 epochs
        (96x96)   : 86.9% accuracy- small_subset/10 epochs 

    EfficientNetV2B1 - Designed for 240x240
        (240x240) : 91% accuracy- small_subset/10 epochs
        (224x224) : 91% accuracy- small_subset/10 epochs
        (224x224) : 91.2% accuracy- small_subset/20 epochs
        (96x96)   : 87.5% accuracy- small_subset/10 epochs 

    *EfficientNetV2B2 - Designed for 260x260
        (260x260) : __% accuracy- small_subset/10 epochs
        (224x224) : 92% accuracy- small_subset/10 epochs
        (96x96)   : 86.3% accuracy- small_subset/10 epochs 

    EfficientNetV2B3 - Designed for 300x300
    (300x300) : __% accuracy- small_subset/10 epochs
    (224x224) : __% accuracy- small_subset/10 epochs
    (96x96)   : __% accuracy- small_subset/10 epochs 

    MobileNetV3Small - Designed for 224x224
        (224x224) : 88% - small_subset/10 epochs
        (224x224) : 88.8% - small_subset/20 epochs
        (96x96)   : 83.6% accuracy - small_subset/10 epochs
        (96x96)   : 84.6% accuracy - small_subset/30 epochs

    MobileNetV3Large - Designed for 224x224
        (224x224) : 91% accuracy - small_subset/10 epochs
        (96x96)   : 87.3% accuracy - small_subset/10 epochs
    
    MobileNetV2 - Designed for 224x224
        (224x224) : __% - small_subset/10 epochs
        (96x96)   : 60% accuracy - small_subset/10 epochs

    NASNetMobile - Designed for 224x224
        (224x224) : 84% accuracy - small_subset/10 epochs
        (96x96)   : 81% accuracy - small_subset/10 epochs