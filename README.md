# 🎬 Simple RNN - IMDB Movie Review Sentiment Analysis

A Deep Learning project that uses a **Simple Recurrent Neural Network (RNN)** to analyze movie reviews and predict whether the sentiment is **Positive** or **Negative**.

The project includes an interactive web application built with **Streamlit**, allowing users to enter their own movie reviews and get real-time sentiment predictions.

## 🚀 Project Overview

Understanding customer opinions and feedback is an important application of Natural Language Processing (NLP).

In this project, I developed a sentiment analysis model using the IMDB movie review dataset. The model processes text data, learns patterns from movie reviews, and predicts the sentiment of an unseen review.

Users can interact with the trained model through a simple and user-friendly Streamlit interface.

## ✨ Features

- **Real-Time Sentiment Prediction:** Analyze movie reviews instantly.
- **Positive or Negative Classification:** Classifies reviews into two sentiment categories.
- **Confidence Score:** Displays the model's prediction confidence.
- **Interactive Web Interface:** Built using Streamlit.
- **Text Preprocessing:** Converts text into numerical sequences.
- **Deep Learning Model:** Uses an Embedding layer and Simple RNN architecture.

## 🧠 Model Architecture

The project uses a Simple Recurrent Neural Network (Simple RNN) for sequence-based text classification.

**Model workflow:**

1. Input movie review
2. Text preprocessing and tokenization
3. Word-to-integer encoding
4. Sequence padding
5. Embedding layer
6. Simple RNN layer
7. Dense output layer
8. Sentiment prediction

The final output is a probability score used to classify the review as positive or negative.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | Deep Learning and model development |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Streamlit | Interactive web application |
| Jupyter Notebook | Model development and experimentation |
| Git & GitHub | Version control and project hosting |

## 📂 Project Structure

```text
Simple-RNN/
│
├── app.py                  # Streamlit web application
├── simple_rnn_model.h5     # Trained RNN model
├── embedding.ipynb         # Embedding experiments
├── prediction.ipynb        # Prediction implementation
├── simpleRNN.ipynb         # Model training and experimentation
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Vinay980/Simple-RNN.git
```

### 2. Navigate to the Project Directory

```bash
cd Simple-RNN
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 💡 How to Use

1. Open the Streamlit application.
2. Enter a movie review in the text input area.
3. Click the prediction button.
4. The model analyzes the review.
5. View the predicted sentiment and confidence score.

## 📊 Dataset

This project uses the **IMDB Movie Reviews Dataset**, a popular dataset for binary sentiment classification.

- 50,000 movie reviews.
- 25,000 training reviews.
- 25,000 testing reviews.
- Two sentiment classes: Positive and Negative.

The dataset is commonly used for learning and experimenting with Natural Language Processing and Deep Learning.

## 🎯 Learning Outcomes

Through this project, I explored and practiced:

- Understanding Recurrent Neural Networks.
- Text preprocessing for NLP.
- Tokenization and word indexing.
- Word embeddings.
- Sequence padding.
- Training a Simple RNN model.
- Making predictions using a trained model.
- Integrating a Deep Learning model with Streamlit.
- Building an interactive machine learning application.

## 🔮 Future Improvements

- Experiment with LSTM and GRU architectures.
- Improve prediction accuracy through hyperparameter tuning.
- Add more NLP preprocessing techniques.
- Compare Simple RNN, LSTM, and GRU performance.
- Deploy the application online.

## 👨‍💻 Author

**Vinay Kumar Mahto**

Frontend Developer | Exploring Deep Learning, Machine Learning, and Generative AI.

- GitHub: [Vinay980](https://github.com/Vinay980)

---

⭐ If you find this project useful, consider giving it a star!
