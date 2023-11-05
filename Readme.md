# ADSenti: Advertisement Sentiment Analysis with LSTM

## Overview

ADSenti is a project that leverages Long Short-Term Memory (LSTM) networks to perform sentiment analysis on advertisement texts. This model classifies advertisement content as positive, negative, or neutral based on its sentiment.

## Data

- The dataset used for this project consists of labeled advertisement texts, where each advertisement has a known sentiment label (positive, negative, or neutral). This data can be collected from various sources, including customer reviews, surveys, and feedback channels.

## Prerequisites

- Python (version X.X)
- Libraries: NumPy, Pandas, Scikit-Learn, Keras (with TensorFlow backend), Matplotlib, etc.

## Project Structure

- `data/`: Directory to store the dataset.
- `scripts/`: Contains Python scripts for data preprocessing, model training, and evaluation.
- `model/`: Store trained LSTM models.
- `README.md`: This readme file.

## Getting Started

1. Clone this repository to your local machine.

2. Create a Python virtual environment (recommended) and install the required libraries using the following command:

   ```shell
   pip install -r requirements.txt
   ```

3. Prepare your dataset of labeled advertisement texts and place it in the `data/` directory.

4. Run the data preprocessing script to clean and tokenize the advertisement texts.

5. Modify the hyperparameters and architecture of the LSTM model in the training script based on your specific dataset and requirements.

6. Train the LSTM model on your dataset using the training script.

7. Evaluate the model's performance on a separate testing set using the evaluation script.

8. Deploy the trained model for sentiment analysis of advertisement texts in your application or use case.

## Usage

Here are some sample commands to run the main scripts:

- To preprocess the data:

  ```shell
  python scripts/preprocess_data.py
  ```

- To train the LSTM model:

  ```shell
  python scripts/train_lstm_model.py
  ```

- To evaluate the model:

  ```shell
  python scripts/evaluate_model.py
  ```

- To make predictions on new advertisement texts:

  ```python
  # Load the trained model
  model = load_model('model/trained_lstm_model.h5')

  # Preprocess and tokenize the new advertisement text
  new_text = "Your new advertisement text here."
  tokenized_text = preprocess_and_tokenize(new_text)

  # Make predictions
  sentiment = model.predict(tokenized_text)
  ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- References to libraries, tools, or datasets used in the project.
