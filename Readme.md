# Happy and Sad Emotion Classification using CNN

This project aims to develop and evaluate a Convolutional Neural Network (CNN) model capable of recognizing human emotions by categorizing images into two distinct classes: "happy" and "sad". Leveraging deep learning techniques, the model classifies facial expressions based on visual features. This approach to automated emotion recognition could enhance applications in human-computer interaction, social robotics, and user experience design.

## Dataset

The dataset consists of happy and sad facial images obtained from the following Kaggle repositories:
- [Yaghoobpoor, 2023](https://www.kaggle.com/datasets/saharnazyaghoobpoor/happy-and-sad-image)
- [Ananthu, 2022](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)

Images are organized into directories: `./happy` and `./sad`.

## Prerequisites

To execute the code, you need to install the following packages:
- Python environment with version `3.x`
- TensorFlow
- scikit-learn
- matplotlib
- seaborn

Installation can be performed using:
```bash
!{sys.executable} -m pip install --upgrade pip setuptools wheel packaging tensorflow scikit-learn seaborn
```

## Project Structure

### 1. Data Loading and Preprocessing

- Utilizes `ImageDataGenerator` for image rescaling and augmentation.
- Splits data into 80% training and 20% validation sets.
- Normalizes pixel values to the range [0, 1].
- Rescales images to 64×64 pixels for consistent input size.
  
### 2. Exploratory Data Analysis (EDA)

- Visualizes class distribution to address potential data imbalance.
- Displays random sample images to illustrate dataset content.

### 3. CNN Model Development

- Builds a CNN using Keras with layers suited to capture complex image features:
  - Multiple Conv2D layers with ReLU activation and Batch Normalization.
  - Pooling layers to reduce spatial dimensions.
  - Fully connected layers culminating in a binary classification output with sigmoid activation.

### 4. Training and Evaluation

- Compiles the model with the Adam optimizer and binary cross-entropy loss.
- Trains the model over multiple epochs, achieving significant improvements in accuracy and reduction in loss.
- Evaluates model performance using a confusion matrix and classification report.

### 5. Deployment and Predictions

- The trained model is saved as `"happy_sad_model.keras"` for reuse.
- Predicts emotions in unseen images using the saved model, providing a confidence score with each classification.

## Conclusion

The CNN model achieved commendable accuracy in classifying "happy" and "sad" emotions, demonstrating high potential for real-world application. By effectively harnessing deep learning for visual feature extraction, this project affirms the utility of CNNs in emotion recognition tasks.

## References

- Russell, S. J., & Norvig, P. (2021). Artificial intelligence: A modern approach (4th ed., Global ed.). Pearson Education.
- TensorFlow. (n.d.). TensorFlow Python API. Retrieved from [TensorFlow API Docs](https://www.tensorflow.org/api_docs/python/tf)

For questions or further information, please refer to the Jupyter Notebook `image_classification.ipynb` included in this repository.