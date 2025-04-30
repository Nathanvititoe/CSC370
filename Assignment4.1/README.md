# Screenshots are located in Assignment4.1/Screenshots folder

## Used this dataset, downloaded as a zip and then added to the project "Assignment4.1/dataset" directory:
    https://www.kaggle.com/datasets/puneet6060/intel-image-classification


## Final Validation Accuracy -- 94.00%




### Testing different base models
---------------------------------------------------------------

#### Final Tests
Chose ConvNeXt_Tiny because it was giving me similar accuracy to much larger, slower pre-trained models but without the size, slow speeds. 

            ------------------------ FINAL ------------------------
                *** ConvNeXt_tiny(224x224) - 94% ***
            -------------------------------------------------------

#### test more augmentation on training only
    baseline - 92.2%
        horizontal flip (0.4) - 92.1%
        horizontal flip (0.5) - 92.2% ADDED
        horizontal flip (0.2) - 91.9%
        randomResizeCrop - 91.3%
        Color jitter(0.4, 0.4, 0.4, 0.2) - 91.73%
        Color jitter(0.4, 0.4, 0.4, 0.2), p=0.5 - %

        Color jitter(0.2, 0.2, 0.2, 0.1) - 92.03%
        Color jitter(0.2, 0.2, 0.2, 0.1), p=0.5 - 92.6% ADDED
        Color jitter(0.2, 0.2, 0.2, 0.1), p=0.25 - 92.4%
        Color jitter(0.2, 0.2, 0.2, 0.1), p=0.75 - 92.2%
        test removing normalize from transforms - 91.87%

    added above changes - baseline = 92.6% (tested in this order)
        testing 150x150 - 93.2%
        .2 dropout(pre bn(96x96)) - 92%
        .1 dropout(pre bn(96x96)) - 92.9% ADDED
        testing 150x150 - 93.4%
        *** testing 224x224 - 94% ***

    ---------------------------------------------------------------
    ConvNeXt_small(96x96) - 91.5% (256 BS) -- no dropout, with Augmentation
    ConvNeXt_small(150x150) - 92.7% (256 BS) -- no dropout, with Augmentation

    ConvNeXt_base(96x96) - 90.5% (256 BS) -- no dropout, with Augmentation
    ConvNeXt_base(150x150) - 92.7% (256 BS) -- no dropout, with Augmentation

    ConvNeXt_large(96x96) - 90.5% (256 BS) -- no dropout, with Augmentation
    ConvNeXt_large(150x150) - 92.7% (256 BS) -- no dropout, with Augmentation

    ConvNeXt_tiny(150x150) - 93.03% (256 BS) -- no dropout, with Augmentation
    ConvNeXt_tiny(224x224) - 93.1 % (256 BS) -- no dropout/ with Augmentation

    ---------------------------------------------------------------
    ConvNeXt_tiny(96x96) - 91.5% (256 BS, 0.3 dropout) -- pre classifier dropout
    ConvNeXt_tiny(96x96) - 92.4% (256 BS, 0.3 dropout) -- pre batchnorm dropout
    ConvNeXt_tiny(96x96) - 92.1% (256 BS, 0.3 dropout) -- start and mid seq dropout
    ConvNeXt_tiny(96x96) - 92.33% (256 BS, 0.3 dropout) -- post relu dropout
    ConvNeXt_tiny(96x96) - 91.5% (256 BS, 0.5 dropout) -- post relu dropout

    ConvNeXt_tiny(96x96) - 91% (64 BS, 0.3 dropout) -- pre batchnorm dropout
    ConvNeXt_tiny(150x150) - 91.4% (256 BS) -- dual 15% dropout/ with Augmentation
    ConvNeXt_tiny(96x96) - 92.1% (256 BS) -- no dropout, no augmentation

    ---------------------------------------------------------------

    EfficientNetB0 (2019)
        (224x224) : 87.7%
        (150x150) : 85.4%

    EfficientNetB3 (2019)
        (224x224) : 87%
        (150x150) : 84.3%

    EfficientNetV2_S (2021)
        (224x224) : 85.4%
        (150x150) : 80.7%

    EfficientNetV2_M (2021)
        (224x224) : 83.5%

    EfficientNetV2_L (2021) - trained on 384x384
        (224x224) : 89.37%
        (384x384) : 91.6%

    Vision Transformer - (2020)
        (224x224) : 92.7%

    convNext_tiny (2022)
        (224x224) BS-128 : 93.00%
        (224x224) BS-256 : 93.23%
        (224x224) BS-256 : 93.5% (0.2 Dropout)
        (224x224) BS-256 : 92.7% (0.2 Dropout) using Adam/not Adamax
        
    convNext_small (2022)
        (224x224) BS-32  : 93.4%
        (224x224) BS-256 : 92.3%
        (224x224) BS-512 : 93.43%
    
    convNext_base (2022)
        (224x224) BS-128 : 93.3%

    convNext_large (2022)
        (224x224) : 93.2%