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
    -- maybe test mobile net lite for speed/accuracy, only available through tensorflowhub api
    -- EfficientNetv1 works better w EfficientNetV2 preprocessing??
    -- EfficientNetv2 works better w EfficientNetv1 preprocesing?? 

    -------------------------------- Final Tests -------------------------------
    EfficientNetB0(224x224) using EfficientNetV2 preprocessing (batch size: 16/10 epochs)
        0.5 dropout/avg pooling - 93.75% accuracy (slight overfit)
        0.5 dropout/max pooling - 92.2% accuracy

        0.7 dropout/avg pooling - 93% accuracy (no overfit)
        0.6 dropout/avg pooling - __% accuracy 

    EfficientNetB0(224x224) using no preprocessing (batch size: 16/20 epochs)
      Apparently preprocessing is built into the model??
        0.5 dropout/avg pooling - __% accuracy (slight overfit)
        0.5 dropout/max pooling - __% accuracy
        0.7 dropout/avg pooling - __% accuracy
    ----------------------------------------------------------------------------
    
    
    EfficientNetB0 using EfficientNetV2 preprocessing (batch size: 16)
        (224x224) : 93.5% accuracy - small_subset/10 epochs
        ***(224x224) : 94.2% accuracy - small_subset/20 epochs***
        (96x96)   : 90% accuracy- small_subset/10 epochs

    EfficientNetB0 using EfficientNetV2 preprocessing (batch size: 8)
        (224x224) : 91.5% accuracy - small_subset/30 epochs w/ augmentation

    EfficientNetV2B0 using EfficientNetV1 preprocessing (batch size: 16)
        (96x96) : 90% accuracy - small_subset/10 epochs

    EfficientNetB0 - Designed for 224x224
        (224x224) : 93.5% accuracy- small_subset/10 epochs 
        (224x224) : 91.5% accuracy- small_subset/10 epochs(RETESTED)
        (96x96)   : 85.4% accuracy- small_subset/10 epochs
        (96x96)   : 89.5% accuracy- small_subset/10 epochs(RETESTED)

    EfficientNetB3 - Designed for 300x300
        (300x300) : 91.2% accuracy- small_subset/10 epochs
        (224x224) : 91.3% accuracy- small_subset/10 epochs
        (96x96)   : 86% accuracy- small_subset/10 epochs   
    
    EfficientNetV2B0 - Designed for 224x224
        (224x224) : 91.3% accuracy- small_subset/10 epochs
        (96x96)   : 86.9% accuracy- small_subset/10 epochs

    EfficientNetV2B1 - Designed for 240x240
        (240x240) : 91% accuracy- small_subset/10 epochs 
        (224x224) : 90% accuracy- small_subset/10 epochs
        (96x96)   : 87.5% accuracy- small_subset/10 epochs  
 
    EfficientNetV2B2 - Designed for 260x260
        (260x260) : 90.2% accuracy- small_subset/10 epochs 
        (224x224) : 90.5% accuracy- small_subset/10 epochs 
        (96x96)   : 86.3% accuracy- small_subset/10 epochs 

    EfficientNetV2B3 - Designed for 300x300
        (300x300) : 92.5% accuracy- small_subset/10 epochs 
        (224x224) : 93.3% accuracy- small_subset/10 epochs 
        (96x96)   : 90.5% accuracy- small_subset/10 epochs 

    MobileNetV3Small - Designed for 224x224
        (224x224) : 88% - small_subset/10 epochs
        (96x96)   : 83.6% accuracy - small_subset/10 epochs

    MobileNetV3Large - Designed for 224x224
        (224x224) : 91% accuracy - small_subset/10 epochs
        (96x96)   : 87.3% accuracy - small_subset/10 epochs
    
    MobileNetV2 - Designed for 224x224
        (224x224) : 62.4% accuracy - small_subset/10 epochs
        (96x96)   : 60% accuracy - small_subset/10 epochs

    NASNetMobile - Designed for 224x224
        (224x224) : 84% accuracy - small_subset/10 epochs
        (96x96)   : 81% accuracy - small_subset/10 epochs