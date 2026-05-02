# Image Caption Generator using Deep Learning

An end-to-end deep learning project for generating captions from images using computer vision and natural language processing techniques.

This project was developed as part of an Artificial Neural Network course project. The main goal is to build a model that takes an image as input and generates a meaningful sentence describing the image.

---

## Project Overview

Image Caption Generation is a task that combines two main fields:

- Computer Vision: to understand the content of the image.
- Natural Language Processing: to generate a sentence that describes the image.

The project follows the Encoder-Decoder architecture:

- Encoder: extracts important features from the image using CNN-based models.
- Decoder: generates the caption word by word using RNN-based or attention-based models.

In this project, multiple deep learning architectures were implemented and compared.

---

## Project Idea

The model receives an image and generates a caption describing it.

Example:

```text
Input Image: people walking on a street
Generated Caption: people walking on pedestrian lane
```

The idea is useful because it can be applied in many real-world applications such as:

- Helping visually impaired people understand images
- Automatic image description
- Image search engines
- Social media image tagging
- Content understanding systems

---

## Dataset

The dataset was collected using web scraping from Unsplash.

The dataset contains images with English captions and Arabic translated captions.

Dataset statistics:

| Item | Count |
|---|---:|
| Images | 10,620 |
| English Captions | 10,620 |
| Arabic Captions | 10,620 |

The dataset contains three main columns:

- Image name
- English caption
- Arabic caption

The scraping process was used to collect images and captions from Unsplash, then the captions were processed and translated into Arabic.

Due to GitHub file size limits, the dataset is not uploaded directly to this repository.

Dataset link:

```text
ADD_DATASET_LINK_HERE
```

---

## Why This Project?

This project is a strong neural network project because it includes:

- Image processing
- Text preprocessing
- Web scraping
- CNN feature extraction
- RNN-based sequence generation
- Attention-based sequence modeling
- Multiple model architectures
- English and Arabic caption generation
- Model training and evaluation
- GUI implementation

It is also suitable for presentation because the input and output are easy to understand visually.

---

## Models Implemented

This project includes five different neural network architectures.

---

### 1. ResNet50 + LSTM

This model uses ResNet50 as the image encoder and LSTM as the text decoder.

#### How it works

- ResNet50 extracts deep visual features from the image.
- The extracted image features are passed to the LSTM decoder.
- The LSTM generates the caption word by word.

#### Why ResNet50?

ResNet50 is a powerful pretrained CNN model. It was trained on a very large image dataset, so it can extract strong image features such as objects, shapes, colors, and visual patterns.

#### Why LSTM?

LSTM is good for sequence generation because it can remember previous words and use them to predict the next word.

#### Benefit

This model is strong and reliable because it uses transfer learning with a pretrained CNN and a sequence model that is suitable for text generation.

#### Implemented Languages

This architecture was implemented for:

- English caption generation
- Arabic caption generation

#### Results

English model:

| Metric | Value |
|---|---:|
| Test Loss | 0.7679 |
| Test Accuracy | 0.8463 |

Arabic model:

| Metric | Value |
|---|---:|
| Test Loss | 0.8792 |
| Test Accuracy | 0.8457 |

---

### 2. VGG16 + GRU

This model uses VGG16 as the image encoder and GRU as the text decoder.

#### How it works

- VGG16 extracts image features.
- GRU receives the image features and caption tokens.
- GRU generates the caption sequence.

#### Why VGG16?

VGG16 is a classic CNN architecture that is simple, clear, and effective for image feature extraction.

#### Why GRU?

GRU is similar to LSTM but simpler and faster. It has fewer gates, so it can train faster while still handling sequential data well.

#### Benefit

This model is useful for comparing a classic pretrained CNN with a lighter recurrent decoder.

---

### 3. CNN + Transformer

This model uses a custom CNN for image feature extraction and a Transformer-style attention mechanism for caption generation.

#### How it works

- A custom CNN extracts visual features from the image.
- Text tokens are converted into embeddings.
- A Multi-Head Attention layer helps the model learn relationships between words and features.
- The model predicts the next words in the caption.

#### Why CNN?

A custom CNN allows us to build the image feature extractor from scratch instead of depending only on pretrained models.

#### Why Transformer?

Transformers are powerful for sequence modeling because they use attention to understand relationships between different parts of the sequence.

#### Benefit

This model is more modern because it uses an attention-based architecture. It helps compare traditional RNN models with Transformer-based models.

---

### 4. CNN + Attention LSTM

This model combines a custom CNN, an attention mechanism, and LSTM.

#### How it works

- CNN extracts image features.
- LSTM processes the caption sequence.
- Attention helps the decoder focus on the most important image features while generating words.
- The final LSTM generates the caption.

#### Why Attention?

Attention improves caption generation because the model does not treat all image features equally. It learns which features are more important for each generated word.

#### Benefit

This model improves the basic CNN + LSTM approach by adding attention, making the decoder more focused and more flexible.

---

### 5. CNN + Vanilla RNN

This model uses a custom CNN with a SimpleRNN decoder.

#### How it works

- A custom CNN extracts image features.
- A SimpleRNN processes the caption tokens.
- The model generates the caption word by word.

#### Why Vanilla RNN?

Vanilla RNN is the simplest recurrent neural network. It is useful as a baseline model.

#### Benefit

This model is important for comparison. It shows how a simpler architecture performs compared to stronger models like LSTM, GRU, Attention, and Transformer.

#### Result

| Metric | Value |
|---|---:|
| Test Loss | 0.8057 |
| Test Accuracy | 0.8472 |

---

## Model Comparison

| Model | Encoder | Decoder | Main Benefit |
|---|---|---|---|
| ResNet50 + LSTM | Pretrained ResNet50 | LSTM | Strong feature extraction and good sequence generation |
| VGG16 + GRU | Pretrained VGG16 | GRU | Simpler and faster than LSTM |
| CNN + Transformer | Custom CNN | Attention-based Transformer | Modern attention-based sequence modeling |
| CNN + Attention LSTM | Custom CNN | Attention + LSTM | Focuses on important image features |
| CNN + Vanilla RNN | Custom CNN | SimpleRNN | Baseline model for comparison |

---

## Preprocessing Pipeline

The preprocessing steps include:

1. Loading the dataset CSV file
2. Reading image names and captions
3. Loading images from the image folder
4. Resizing images to 224x224
5. Converting images from BGR to RGB
6. Cleaning English captions
7. Cleaning Arabic captions
8. Adding `<start>` and `<end>` tokens
9. Tokenizing captions
10. Padding sequences to a maximum length of 30
11. Splitting the dataset into training, validation, and testing sets
12. Creating target sequences by shifting captions
13. Applying data augmentation

---

## Data Augmentation

Data augmentation was used to improve model generalization.

The augmentation techniques include:

- Random horizontal flip
- Random rotation
- Random zoom
- Random brightness
- Random contrast

These techniques help the model learn better by showing it slightly different versions of the same images.

---

## Web Scraping

The project includes a `scraping.py` file.

The purpose of this file is to collect image-caption data from Unsplash.

The scraping module can be used to:

- Collect image URLs
- Download images
- Save image names
- Save captions
- Store the final data in CSV format

This makes the project more complete because it includes a way to prepare or expand the dataset.

---

## GUI

The project includes a simple graphical user interface using Tkinter.

The GUI allows the user to:

- Upload an image
- Preview the selected image
- Generate or display a caption
- Make the project easier to demonstrate during presentation

The GUI file is located at:

```text
src/gui.py
```

---

## Project Structure

```text
Image-caption-generator-/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── image_caption_generator_training.ipynb
│
├── src/
│   ├── scraping.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── inference.py
│   │
│   ├── model_resnet50_lstm.py
│   ├── model_vgg16_gru.py
│   ├── model_cnn_transformer.py
│   ├── model_cnn_attention_lstm.py
│   ├── model_cnn_vanilla_rnn.py
│   │
│   └── gui.py
│
├── results/
│   └── model_comparison.md
│
├── models/
│   └── README.md
│
└── data/
    └── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ahmedwalidahmad-debug/Image-caption-generator-.git
```

Move to the project folder:

```bash
cd Image-caption-generator-
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## How to Run

Train the model:

```bash
python src/train.py
```

Run inference:

```bash
python src/inference.py
```

Run the GUI:

```bash
python src/gui.py
```

---

## Requirements

The main libraries used in this project are:

- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- Deep Translator
- BeautifulSoup
- Requests
- Pillow

---

## Results Summary

| Model | Language | Test Loss | Test Accuracy |
|---|---:|---:|---:|
| ResNet50 + LSTM | English | 0.7679 | 0.8463 |
| ResNet50 + LSTM | Arabic | 0.8792 | 0.8457 |
| CNN + Vanilla RNN | English | 0.8057 | 0.8472 |

---

## Future Improvements

Possible future improvements include:

- Training on a larger dataset
- Using BLEU score for caption evaluation
- Adding beam search decoding
- Fine-tuning pretrained CNN models
- Improving Arabic caption generation
- Improving the GUI
- Deploying the project as a web application
- Adding more evaluation metrics

---

## Academic Integrity

This project was implemented for educational purposes as part of a Neural Network course project.

The project includes:

- Dataset preparation
- Web scraping from Unsplash
- Image preprocessing
- Text preprocessing
- Multiple neural network architectures
- Training and evaluation
- Result comparison
- GUI implementation

---

## Author

Ahmed Walid

---

## License

This project is for educational purposes.