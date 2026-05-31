# BirdCLEF - Acoustic Species Identification in the Pantanal, South America

This project aims to predict 235 different birds and amphibious species from audio recordings recorded by different biology research groups.
We code and train from scratch, leveraging state-of-the art AI techniques such as Model Stacking/Blending, usage of priors (bayesian probab.) as well as audio augmentation pipelines.


## 1. Architecture 🤖
First step is to convert the audio to a MEL spectrogram using `librosa.feature.melspectrogram`.

In order to obtain the features we do use a simple image encoder `efficientnet_b0` implemented by the `timm` library, using pre-trained weights for the most part of the training. in the last step, the top layer will be unfrozen - this flexibility should allow the encoder to focus on features that correlate with the predicted class.

Lastly, a 2-layer Attention pooling block implemented by `torch.nn` will map the feature image to the corresponding class. Its selection mechanism allows to ignore silence and focus on the important parts (the sparse bird calls). In future, a self-attention block could help relate bird calls ,that are separated in time, to each other, capturing long-range dependencies.


## 3. Training
We use the typical overfitting procedure: Overfit a tiny dataset of 4 audio clips to ensure the data pipeline is working. Once this is done, we progressively add more data and help the model to generalize.
4. Priors
5. Stacking
