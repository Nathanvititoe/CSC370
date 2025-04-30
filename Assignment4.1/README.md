# Screenshots are located in Assignment4.1/Screenshots folder

## Used this dataset, downloaded as a zip and then added to the project "Assignment4.1/dataset" directory:
    https://www.kaggle.com/datasets/puneet6060/intel-image-classification

### Testing different base models
---------------------------------------------------------------

#### Final Tests
    ------------------------ FINAL ------------------
    *** ConvNeXt_tiny(150x150) - % (256 BS) -- no dropout / with Augmentation ***
    -------------------------------------------------

    ConvNeXt_tiny(150x150) - 93.03% (256 BS) -- no dropout, with Augmentation

    ConvNeXt_tiny(224x224) - 93.1 % (256 BS) -- no dropout/ with Augmentation

    ----------------------------------------------
    ConvNeXt_tiny(96x96) - 91.5% (256 BS, 0.3 dropout) -- pre classifier dropout
    ConvNeXt_tiny(96x96) - 92.4% (256 BS, 0.3 dropout) -- pre batchnorm dropout
    ConvNeXt_tiny(96x96) - 92.1% (256 BS, 0.3 dropout) -- start and mid seq dropout
    ConvNeXt_tiny(96x96) - 92.33% (256 BS, 0.3 dropout) -- post relu dropout
    ConvNeXt_tiny(96x96) - 91.5% (256 BS, 0.5 dropout) -- post relu dropout

    ConvNeXt_tiny(96x96) - 91% (64 BS, 0.3 dropout) -- pre batchnorm dropout
    ConvNeXt_tiny(150x150) - 91.4% (256 BS) -- dual 15% dropout/ with Augmentation
    ---------------------------------------------

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