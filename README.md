## Methodology

The development of the image captioning model involved the following key stages:

### 1. Data Loading and Preprocessing
* **Image and Caption Loading:** Raw images from the Flickr8k dataset and their corresponding textual captions were loaded.
* **Image Preprocessing:** Necessary preprocessing steps were applied to the images to prepare them for feature extraction. This typically includes resizing, normalization, and formatting suitable for the chosen CNN model.
* **Text Preprocessing and Vectorization:** The textual captions underwent extensive preprocessing. This involved tokenization, cleaning (e.g., removing punctuation, converting to lowercase), building a vocabulary, and converting captions into numerical sequences (vectorization). Special tokens (e.g., start-of-sequence, end-of-sequence) were incorporated. The rationale behind each preprocessing step was considered to ensure optimal model performance.

### 2. Feature Extraction
* A **pre-trained Convolutional Neural Network (CNN)** (e.g., VGG, ResNet, InceptionV3) was employed as an image feature extractor. The output from one of the CNN's later layers (before the final classification layer) was used as a fixed-length vector representation (feature vector) for each image.

### 3. Dataset Preparation
* The extracted image feature vectors and the preprocessed, vectorized captions were combined to create the final dataset for the caption generation model.
* This dataset was then split into **training and testing sets** to facilitate model training and unbiased evaluation.

### 4. Model Architecture: img2seq GRU
An encoder-decoder architecture using Gated Recurrent Units (GRUs) was designed and implemented for generating captions:
* **Encoder:** The encoder part of the model processes the image feature vectors. In some designs, this might involve a dense layer to transform the image features into an initial hidden state for the decoder.
* **Decoder:** The decoder is a GRU-based recurrent neural network. It takes the image information (from the encoder) and the previously generated words in the caption as input to predict the next word in the sequence.
    * An **embedding layer** was used to convert input word indices into dense vector representations.
    * One or more **GRU layers** formed the core of the decoder, capturing temporal dependencies in the caption sequence.
    * A **masking layer** might have been used to handle variable-length sequences during training.
    * A final dense layer with a softmax activation function was used to predict the probability distribution over the vocabulary for the next word.

### 5. Training and Evaluation
* The img2seq GRU model was trained on the prepared training set, optimizing for a suitable loss function (e.g., categorical cross-entropy).
* The model's performance was evaluated on the test set using appropriate metrics for sequence generation tasks (e.g., BLEU score, METEOR, ROUGE, CIDEr).

### 6. Inference and Demonstration
* A dedicated inference function was created to generate captions for new, unseen images using the trained model.
* This function was tested by providing five different images taken with a mobile camera. The generated captions were displayed alongside their respective images to demonstrate the model's captioning capabilities in a real-world scenario.
