# 🛰️ SkyCaptions - Satellite Image Captioning using BiLSTMs and ViT Feature Extractor

SkyCaptions generates captions for satellite images using Bidirectional Long Short-Term Memory (BiLSTM) networks and the Vision Transformer (ViT) feature extractor. This combination allows for effective extraction of image features and sequential modeling of captions.

![Project Preview](/path/to/project-preview.png)

## Table of Contents

- [🚀 Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [✨ Features](#features)
- [📚 Dataset](#dataset)
- [🎰 Model Summary](#model-summary)
- [🤝 Contributing](#contributing)
- [📝 License](#license)
- [🙌 Acknowledgments](#acknowledgments)

## Getting Started

These instructions will help you get a copy of the SkyCaptions project up and running on your local machine for development and testing purposes.

### Prerequisites

To run this project, you'll need the following:

- [Python](https://www.python.org/downloads/) installed on your system.
- [TensorFlow](https://www.tensorflow.org/install) or [PyTorch](https://pytorch.org/get-started/locally/) (depending on your implementation preference).

## Features
* ViT Feature Extraction: Extracts high-level features from satellite images using the Vision Transformer (ViT).
* BiLSTM Captioning: Generates captions for satellite images using Bidirectional Long Short-Term Memory (BiLSTM) networks.
* Training and Evaluation: Provides scripts for training the model and evaluating its performance on test datasets.
* Data Visualization: Visualizes the results with the generated captions overlayed on the images.

## Dataset
This project uses the satellite image caption generation dataset from Kaggle.
You can find and download the dataset from the following link: [dataset](https://www.kaggle.com/datasets/tomtillo/satellite-image-caption-generation)

## Model Summary
```bash
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
input_2 (InputLayer)        [(None, 768)]                0         []                            
                                                                                                  
dense (Dense)               (None, 512)                  393728    ['input_2[0][0]']             
                                                                                                  
input_3 (InputLayer)        [(None, 53)]                 0         []                            
                                                                                                  
reshape_1 (Reshape)         (None, 1, 512)               0         ['dense[0][0]']               
                                                                                                  
embedding (Embedding)       (None, 53, 512)              1381376   ['input_3[0][0]']             
                                                                                                  
concatenate (Concatenate)   (None, 54, 512)              0         ['reshape_1[0][0]',           
                                                                     'embedding[0][0]']           
                                                                                                  
bidirectional (Bidirection  (None, 512)                  1574912   ['concatenate[0][0]']         
al)                                                                                              
                                                                                                  
dropout (Dropout)           (None, 512)                  0         ['bidirectional[0][0]']       
                                                                                                  
add (Add)                   (None, 512)                  0         ['dropout[0][0]',             
                                                                     'dense[0][0]']               
                                                                                                  
dense_1 (Dense)             (None, 128)                  65664     ['add[0][0]']                 
                                                                                                  
dropout_1 (Dropout)         (None, 128)                  0         ['dense_1[0][0]']             
                                                                                                  
dense_2 (Dense)             (None, 2698)                 348042    ['dropout_1[0][0]']           
                                                                                                  
==================================================================================================
Total params: 3763722 (14.36 MB)
Trainable params: 3763722 (14.36 MB)
Non-trainable params: 0 (0.00 Byte)

```
## Contributing
Contributions are welcome! Please feel free to open a pull request or report any issues.

## License
This project is licensed under the MIT License

## Acknowledgments
* Thanks to the open-source community for providing valuable resources and tools.
* Special thanks to the developers of [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) and [BiLSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) for their foundational work.
* Thanks to [Tom Tillo](https://www.kaggle.com/tomtillo) for providing the [Satellite Image Caption Generation Dataset on Kaggle](https://www.kaggle.com/datasets/tomtillo/satellite-image-caption-generation).
