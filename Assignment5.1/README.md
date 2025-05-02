# Dataset : 
https://www.kaggle.com/datasets/chrisfilo/urbansound8k


# Functioning: 
    YAMNet:
        - takes in raw audio waveforms and converts them into a log-mel spectrogram (visualization of how frequency changes over time).
        
        - uses that spectrogram and feeds it through a Convolutional Neural Network (CNN) that was pre-trained on "Audioset" (large dataset)
        
        - outputs a set of feature vectors with input shape (1024)

    Custom:
        - Remove the YAMNet classifier (top layer)

        - Create a custom classifier (top layer) and use the YAMNet output as input (input_shape=1024)

            -  Single Convolutional Layer 
                    -Dense Layer (256 neurons)
                    -ReLu activation function
            
            - Output Layer with output_shape as the number of classes we are trying to classify (using softmax function)
                 



## Resources :
https://www.geeksforgeeks.org/audio-classification-using-googles-yamnet/

https://www.kaggle.com/code/satoru90/urban-sound-classification

https://stackoverflow.com/questions/55977956/how-to-classify-sound-using-fft-and-neural-network-should-i-use-cnn-or-rnn

https://www.kaggle.com/code/pulkit12dhingra/urban-8k-tensorflow-audio-classification

https://www.kaggle.com/code/christianlillelund/classify-mnist-audio-using-spectrograms-keras-cnn

https://www.tensorflow.org/tutorials/audio/transfer_learning_audio

https://learn.microsoft.com/en-us/training/modules/intro-audio-classification-tensorflow/

https://tqdm.github.io/

### YAMNet & Tensorflow specific -- Audio Classification
- [Audio Classification using Google’s YAMNet – GeeksForGeeks](https://www.geeksforgeeks.org/audio-classification-using-googles-yamnet/)
- [Transfer Learning for Audio – TensorFlow Tutorial](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio)
- [Intro to Audio Classification with TensorFlow – Microsoft Learn](https://learn.microsoft.com/en-us/training/modules/intro-audio-classification-tensorflow/)

### Examples of Same Dataset
- [Urban Sound Classification – Satoru90 (Kaggle)](https://www.kaggle.com/code/satoru90/urban-sound-classification)
- [Urban8K TensorFlow Audio Classifier – Pulkit Dhingra (Kaggle)](https://www.kaggle.com/code/pulkit12dhingra/urban-8k-tensorflow-audio-classification)

### General Audio Classification
- [How to Classify Sound Using FFT and Neural Networks (StackOverflow)](https://stackoverflow.com/questions/55977956/how-to-classify-sound-using-fft-and-neural-network-should-i-use-cnn-or-rnn)
- [Classify MNIST Audio using Spectrograms + Keras CNN – Christian Lillelund (Kaggle)](https://www.kaggle.com/code/christianlillelund/classify-mnist-audio-using-spectrograms-keras-cnn)

### For Loading UI
- [TQDM – Progress Bars for Python](https://tqdm.github.io/)