# Sentiment Analysis with Logistic Regression

This repository contains a simple sentiment analysis implementation using Logistic Regression and TF-IDF features. The goal of this project is to understand the fundamental concepts of sentiment analysis and logistic regression, rather than achieving high accuracy.

## Files

- `sentiment_analysis.py`: Contains the code for sentiment analysis, including data preprocessing, model training, evaluation, and prediction.

## How It Works

1. **Data Preparation**: The dataset contains a list of text samples with associated sentiment labels (1 for positive, 0 for negative).

2. **Text Vectorization**: The text data is transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

3. **Model Training**: A Logistic Regression model is trained on the TF-IDF features of the training data.

4. **Evaluation**: The model's performance is evaluated on the test data using accuracy and a classification report.

5. **Prediction**: The model predicts the sentiment of new text samples.

## Usage

1. **Install Dependencies**: Ensure you have the required libraries installed. You can install them using pip:
    ```bash
    pip install nltk scikit-learn
    ```

2. **Run the Code**: Execute the script to see the sentiment analysis in action.
    ```bash
    python SentimentAnalysis.py
    ```

## Notes

- The current model's accuracy is approximately 0.6. This project is intended for educational purposes to understand the underlying logic of sentiment analysis, not to achieve high accuracy.
- Feel free to modify the dataset, model parameters, or text vectorization techniques to experiment and improve the model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
