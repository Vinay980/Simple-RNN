
import streamlit as st
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sentiment AI - Movie Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MAX_LENGTH = 500


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.html("""
<style>

.stApp {
    background: #080d1b;
    color: #f8fafc;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

.block-container {
    max-width: 1150px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Typography */

h1, h2, h3, h4 {
    color: #f8fafc !important;
}

p, li {
    color: #aebbd0;
    line-height: 1.7;
}

/* Navigation */

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 5px;
    border-bottom: 1px solid #27334c;
    margin-bottom: 30px;
}

.brand {
    font-size: 25px;
    font-weight: 850;
    color: white;
}

.brand span {
    color: #a78bfa;
}

.nav-links a {
    color: #cbd5e1;
    text-decoration: none;
    margin-left: 20px;
    font-size: 14px;
}

.nav-links a:hover {
    color: #a78bfa;
}

/* Hero */

.hero {
    text-align: center;
    padding: 65px 15px 45px;
}

.hero-badge {
    display: inline-block;
    background: #211a42;
    color: #c4b5fd;
    border: 1px solid #51418a;
    padding: 9px 18px;
    border-radius: 30px;
    font-size: 13px;
    margin-bottom: 20px;
}

.hero h1 {
    font-size: clamp(38px, 6vw, 64px);
    font-weight: 850;
    line-height: 1.15;
    background: linear-gradient(
        90deg,
        #c4b5fd,
        #818cf8,
        #38bdf8
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    max-width: 700px;
    margin: 20px auto;
    font-size: 17px;
    color: #aebbd0;
}

.hero-note {
    color: #94a3b8;
    font-size: 13px;
}

/* Section headings */

.section-heading {
    text-align: center;
    padding: 40px 10px 20px;
}

.section-heading h2 {
    font-size: 32px;
    font-weight: 800;
}

.section-heading p {
    color: #94a3b8;
}

/* Cards */

.info-card {
    background: linear-gradient(145deg, #17223a, #101729);
    border: 1px solid #2d3b59;
    border-radius: 16px;
    padding: 24px;
    min-height: 175px;
    margin-bottom: 15px;
}

.info-card h3 {
    font-size: 19px;
    margin-bottom: 12px;
}

.info-card p {
    font-size: 14px;
}

.card-icon {
    font-size: 30px;
    margin-bottom: 12px;
}

/* Analyzer */

.analyzer-card {
    background: linear-gradient(145deg, #17223a, #11182a);
    border: 1px solid #39486b;
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 15px;
}

.analyzer-title {
    color: white;
    font-size: 25px;
    font-weight: 800;
}

.analyzer-description {
    color: #94a3b8;
    font-size: 14px;
}

/* Input */

textarea {
    background: #0c1425 !important;
    color: white !important;
    border: 1px solid #39486b !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] > div {
    background: #111c32;
    border-color: #39486b;
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #7c3aed, #4f46e5);
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 700;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border: 1px solid #a78bfa;
}

/* Results */

.result-card {
    background: #141f35;
    border: 1px solid #39486b;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}

.positive-result {
    color: #4ade80;
    font-size: 30px;
    font-weight: 800;
}

.negative-result {
    color: #fb7185;
    font-size: 30px;
    font-weight: 800;
}

.result-description {
    color: #94a3b8;
    font-size: 14px;
}

/* Workflow */

.step-card {
    background: #131e33;
    border: 1px solid #34415f;
    border-radius: 15px;
    padding: 24px;
    min-height: 210px;
    margin-bottom: 15px;
}

.step-icon {
    font-size: 30px;
    margin-bottom: 12px;
}

/* Technology */

.tech-card {
    background: #17223a;
    border: 1px solid #34415f;
    border-radius: 15px;
    text-align: center;
    padding: 22px 10px;
    min-height: 145px;
    margin-bottom: 15px;
}

.tech-icon {
    font-size: 32px;
}

.tech-card h3 {
    font-size: 18px;
}

/* Footer */

.footer {
    text-align: center;
    border-top: 1px solid #27334c;
    margin-top: 60px;
    padding: 35px 10px;
    color: #94a3b8;
    font-size: 13px;
}

/* Mobile */

@media(max-width: 700px) {

    .hero {
        padding-top: 35px;
    }

    .hero h1 {
        font-size: 38px;
    }

    .nav-links a {
        margin-left: 7px;
        font-size: 11px;
    }

    .info-card,
    .step-card {
        min-height: auto;
    }

}

</style>
""")


# =========================================================
# 3. LOAD MODEL AND IMDB WORD INDEX
# =========================================================

@st.cache_resource
def load_sentiment_model():

    return load_model(
        "simple_rnn_model.h5"
    )


@st.cache_resource
def load_word_index():

    return imdb.get_word_index()


try:

    model = load_sentiment_model()
    word_index = load_word_index()
    model_error = None

except Exception as error:

    model = None
    word_index = None
    model_error = str(error)


# =========================================================
# 4. TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    # Convert text into lowercase words
    words = text.lower().split()

    # Convert words into IMDB numerical tokens
    encoded_review = [
        word_index[word] + 3 if word in word_index else 2
        for word in words
    ]

    # Pad sequence to 500 tokens
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=MAX_LENGTH
    )

    return padded_review


# =========================================================
# 5. PREDICTION FUNCTION
# =========================================================

def predict_sentiment(review):

    processed_input = preprocess_text(review)

    prediction = model.predict(
        processed_input,
        verbose=0
    )

    score = float(prediction[0][0])

    if score > 0.5:

        sentiment = "Positive"
        confidence = score

    else:

        sentiment = "Negative"
        confidence = 1 - score

    return sentiment, confidence, score


# =========================================================
# 6. NAVIGATION BAR
# =========================================================

st.html("""
<div class="navbar">

    <div class="brand">
        ✦ Sentiment<span>AI</span>
    </div>

    <div class="nav-links">
        <a href="#home">Home</a>
        <a href="#analyzer">Analyzer</a>
        <a href="#how-it-works">How It Works</a>
        <a href="#about">About</a>
        <a href="#technology">Technology</a>
    </div>

</div>
""")


# =========================================================
# 7. HERO SECTION
# =========================================================

st.html("""
<div class="hero" id="home">

    <div class="hero-badge">
        ✨ AI POWERED · DEEP LEARNING · NLP
    </div>

    <h1>
        Understand the Emotion<br>
        Behind Every Review.
    </h1>

    <p>
        Turn movie reviews into meaningful insights.
        Our AI-powered sentiment analyzer uses Deep Learning
        to understand whether a movie review expresses
        a positive or negative opinion.
    </p>

    <div class="hero-note">
        Simple to use. Powered by Artificial Intelligence.
    </div>

</div>
""")


# =========================================================
# 8. FEATURES
# =========================================================

feature1, feature2, feature3 = st.columns(3)

with feature1:

    st.html("""
    <div class="info-card">
        <div class="card-icon">🧠</div>
        <h3>Deep Learning</h3>
        <p>
            Powered by a trained Simple Recurrent
            Neural Network.
        </p>
    </div>
    """)

with feature2:

    st.html("""
    <div class="info-card">
        <div class="card-icon">💬</div>
        <h3>Text Understanding</h3>
        <p>
            Analyzes written movie reviews using
            Natural Language Processing.
        </p>
    </div>
    """)

with feature3:

    st.html("""
    <div class="info-card">
        <div class="card-icon">⚡</div>
        <h3>Instant Prediction</h3>
        <p>
            Get a positive or negative prediction
            in seconds.
        </p>
    </div>
    """)


# =========================================================
# 9. SENTIMENT ANALYZER
# =========================================================

st.html("""
<div class="section-heading" id="analyzer">

    <h2>Try the Sentiment Analyzer</h2>

    <p>
        Enter a movie review and let our AI analyze its sentiment.
        No technical knowledge required.
    </p>

</div>
""")

st.html("""
<div class="analyzer-card">

    <div class="analyzer-title">
        🎬 Movie Review Analyzer
    </div>

    <p class="analyzer-description">
        Write your thoughts about a movie below.
        Our AI model will classify your review
        as positive or negative.
    </p>

</div>
""")


# Sample reviews

SAMPLE_REVIEWS = {

    "Select a sample review": "",

    "😊 Positive - Amazing movie": (
        "This movie was absolutely fantastic! "
        "The acting was brilliant, the story was engaging "
        "and every moment was enjoyable."
    ),

    "😞 Negative - Terrible movie": (
        "This was one of the worst movies I have ever watched. "
        "The story was boring, the acting was terrible "
        "and I completely regret watching it."
    ),

    "⭐ Positive - Great experience": (
        "An outstanding movie with excellent performances "
        "and a beautiful storyline. I highly recommend it."
    ),

    "👎 Negative - Disappointing experience": (
        "I expected much more from this movie. "
        "The plot was confusing and the characters "
        "were poorly developed."
    )
}


def update_sample():

    selected = st.session_state.sample_review

    st.session_state.movie_review = SAMPLE_REVIEWS[selected]


if "movie_review" not in st.session_state:

    st.session_state.movie_review = ""


if "sample_review" not in st.session_state:

    st.session_state.sample_review = "Select a sample review"


st.selectbox(
    "Try an example review",
    options=list(SAMPLE_REVIEWS.keys()),
    key="sample_review",
    on_change=update_sample
)


# Review form

with st.form("review_form"):

    user_input = st.text_area(
        "Your Movie Review",
        key="movie_review",
        height=170,
        placeholder=(
            "Example: This movie was amazing! "
            "The acting was brilliant and the story was fantastic..."
        )
    )

    submitted = st.form_submit_button(
        "✨ Analyze My Review",
        use_container_width=True
    )


# =========================================================
# 10. DISPLAY PREDICTION
# =========================================================

if submitted:

    if not user_input.strip():

        st.warning(
            "Please enter a movie review before analyzing."
        )

    elif model is None:

        st.error(
            "The trained model could not be loaded. "
            "Please check simple_rnn_imdb_relu.h5."
        )

    else:

        try:

            with st.spinner(
                "🧠 AI is analyzing your movie review..."
            ):

                sentiment, confidence, raw_score = (
                    predict_sentiment(user_input)
                )

            st.session_state.analysis_result = {

                "sentiment": sentiment,
                "confidence": confidence,
                "score": raw_score

            }

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# Show result

if "analysis_result" in st.session_state:

    result = st.session_state.analysis_result

    sentiment = result["sentiment"]
    confidence = result["confidence"]
    raw_score = result["score"]

    st.html("""
    <div class="section-heading">
        <h2>Your Analysis Result</h2>
        <p>Here is what our AI thinks about your review.</p>
    </div>
    """)

    if sentiment == "Positive":

        st.html("""
        <div class="result-card">

            <div style="font-size:55px;">😊</div>

            <div class="positive-result">
                Positive Sentiment
            </div>

            <p class="result-description">
                Your review expresses a positive opinion.
            </p>

        </div>
        """)

    else:

        st.html("""
        <div class="result-card">

            <div style="font-size:55px;">😞</div>

            <div class="negative-result">
                Negative Sentiment
            </div>

            <p class="result-description">
                Your review expresses a negative opinion.
            </p>

        </div>
        """)

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Raw Prediction Score",
            f"{raw_score:.4f}"
        )

    st.write("Confidence visualization")

    st.progress(confidence)

    st.caption(
        "Confidence is derived from the model's output. "
        "It is not a measurement of actual prediction accuracy."
    )

    with st.expander("Understand your prediction"):

        st.write(f"**Sentiment:** {sentiment}")

        st.write(f"**Raw model score:** {raw_score:.4f}")

        st.write(
            "**Decision rule:** Scores greater than 0.5 "
            "are Positive. Scores at or below 0.5 are Negative."
        )

        st.write(
            f"**Model confidence:** {confidence * 100:.2f}%"
        )


# =========================================================
# 11. HOW IT WORKS
# =========================================================

st.html("""
<div class="section-heading" id="how-it-works">

    <h2>How Does It Work?</h2>

    <p>
        From your written review to an AI prediction,
        here is what happens behind the scenes.
    </p>

</div>
""")

step1, step2, step3 = st.columns(3)

with step1:

    st.html("""
    <div class="step-card">

        <div class="step-icon">📝</div>

        <h3>01. Text Input</h3>

        <p>
            You enter a movie review in natural language.
            The application receives your text as input.
        </p>

    </div>
    """)

with step2:

    st.html("""
    <div class="step-card">

        <div class="step-icon">⚙️</div>

        <h3>02. Text Processing</h3>

        <p>
            Words are converted into numerical tokens
            using the IMDB vocabulary. The sequence
            is padded to 500 tokens.
        </p>

    </div>
    """)

with step3:

    st.html("""
    <div class="step-card">

        <div class="step-icon">🤖</div>

        <h3>03. AI Prediction</h3>

        <p>
            The trained RNN model processes the sequence
            and predicts positive or negative sentiment.
        </p>

    </div>
    """)


# =========================================================
# 12. WHAT IS SENTIMENT ANALYSIS?
# =========================================================

st.html("""
<div class="section-heading" id="about">

    <h2>What Is Sentiment Analysis?</h2>

    <p>
        Understanding the technology behind this project.
    </p>

</div>

<div class="info-card">

    <h3>💡 Understanding Sentiment Analysis</h3>

    <p>
        Sentiment Analysis is a Natural Language Processing
        (NLP) technique that helps computers identify opinions,
        emotions and attitudes expressed in text.
    </p>

    <p>
        In this project, sentiment analysis determines
        whether a movie review expresses a positive
        or negative opinion.
    </p>

    <p>
        For example:
    </p>

    <p style="color:#4ade80;">
        "The movie was amazing and I loved it." → Positive
    </p>

    <p style="color:#fb7185;">
        "The movie was boring and disappointing." → Negative
    </p>

</div>
""")


# =========================================================
# 13. WHAT IS SIMPLE RNN?
# =========================================================

st.subheader("🧠 What Is a Simple Recurrent Neural Network?")

rnn1, rnn2 = st.columns([1.3, 1])

with rnn1:

    st.html("""
    <div class="info-card">

        <h3>Understanding RNN</h3>

        <p>
            A Recurrent Neural Network (RNN) is a type of
            artificial neural network designed to work
            with sequential data.
        </p>

        <p>
            Unlike a traditional feedforward neural network,
            an RNN maintains a hidden state that carries
            information from previous time steps.
        </p>

        <p>
            This makes RNNs useful for processing text,
            speech and time-series data.
        </p>

        <h3>Why use RNN?</h3>

        <ul>
            <li>Processes words as a sequence.</li>
            <li>Maintains information from previous words.</li>
            <li>Learns patterns from text.</li>
            <li>Supports sentiment classification.</li>
        </ul>

    </div>
    """)

with rnn2:

    st.html("""
    <div class="info-card">

        <h3>Example: Processing a Review</h3>

        <p style="color:#c4b5fd;">
            "This movie was amazing"
        </p>

        <hr>

        <p>Word 1 → This</p>
        <p>Word 2 → movie</p>
        <p>Word 3 → was</p>
        <p>Word 4 → amazing</p>

        <hr>

        <p>
            The RNN processes the word sequence
            and uses learned patterns to make
            a sentiment prediction.
        </p>

    </div>
    """)


# =========================================================
# 14. TECHNOLOGIES USED
# =========================================================

st.html("""
<div class="section-heading" id="technology">

    <h2>Technologies Behind This Project</h2>

    <p>
        Built using popular Python and Deep Learning technologies.
    </p>

</div>
""")

tech1, tech2, tech3, tech4 = st.columns(4)

tech_data = [

    ("🐍", "Python", "Programming language"),

    ("🧠", "TensorFlow", "Deep Learning framework"),

    ("⚡", "Keras", "Neural network API"),

    ("🚀", "Streamlit", "Web application framework")

]

for col, (icon, name, description) in zip(
    [tech1, tech2, tech3, tech4],
    tech_data
):

    with col:

        st.html(f"""
        <div class="tech-card">

            <div class="tech-icon">{icon}</div>

            <h3>{name}</h3>

            <p>{description}</p>

        </div>
        """)


# =========================================================
# 15. IMDB DATASET
# =========================================================

st.subheader("📚 About the IMDB Dataset")

st.html("""
<div class="info-card">

    <h3>Internet Movie Database (IMDB)</h3>

    <p>
        This project uses the IMDB movie review dataset
        available through TensorFlow Keras.
    </p>

    <p>
        The standard Keras IMDB dataset contains 50,000
        movie reviews, divided into 25,000 training reviews
        and 25,000 testing reviews.
    </p>

    <p>
        Each review has a binary sentiment label:
        Positive or Negative.
    </p>

    <p>
        The dataset provides a word index that maps words
        to numerical values so they can be processed
        by a neural network.
    </p>

</div>
""")


# =========================================================
# 16. REAL WORLD APPLICATIONS
# =========================================================

st.html("""
<div class="section-heading">

    <h2>Real-World Applications</h2>

    <p>
        Sentiment analysis can help businesses understand
        what people think and feel.
    </p>

</div>
""")

app1, app2 = st.columns(2)

with app1:

    st.html("""
    <div class="info-card">

        <h3>🛒 E-commerce Reviews</h3>

        <p>
            Analyze customer reviews to understand
            whether people are satisfied with products.
        </p>

    </div>

    <div class="info-card">

        <h3>📱 Social Media Monitoring</h3>

        <p>
            Understand public opinions about products,
            services and brands.
        </p>

    </div>
    """)

with app2:

    st.html("""
    <div class="info-card">

        <h3>🏢 Customer Feedback</h3>

        <p>
            Help businesses understand customer experiences
            and identify areas for improvement.
        </p>

    </div>

    <div class="info-card">

        <h3>🎬 Movie Reviews</h3>

        <p>
            Analyze audience opinions and understand
            whether viewers liked a particular movie.
        </p>

    </div>
    """)


# =========================================================
# 17. FAQ
# =========================================================

st.html("""
<div class="section-heading">

    <h2>Frequently Asked Questions</h2>

    <p>
        Everything you need to know about this application.
    </p>

</div>
""")

with st.expander("What does this website do?"):

    st.write(
        "This website analyzes movie reviews and predicts "
        "whether the expressed sentiment is positive or negative."
    )

with st.expander("Do I need to create an account?"):

    st.write(
        "No. You can use the sentiment analyzer without "
        "creating an account."
    )

with st.expander("Which AI model does this website use?"):

    st.write(
        "It uses a pre-trained Simple Recurrent Neural Network "
        "developed using TensorFlow and Keras."
    )

with st.expander("What does the confidence score mean?"):

    st.write(
        "The confidence score is derived from the model's "
        "sigmoid output. It represents how strongly the model "
        "favors its predicted class, not guaranteed accuracy."
    )

with st.expander("Can it analyze other languages?"):

    st.write(
        "The model is trained on English IMDB movie reviews "
        "and is intended for English-language input."
    )

with st.expander("Can it understand sarcasm?"):

    st.write(
        "Not reliably. A basic RNN sentiment classifier "
        "can struggle with sarcasm, irony and complex context."
    )


# =========================================================
# 18. MODEL STATUS
# =========================================================

if model_error:

    st.warning(
        "The informational website is available, but the "
        "sentiment analyzer could not load its model. "
        "Check that simple_rnn_imdb_relu.h5 exists in "
        "the project directory."
    )


# =========================================================
# 19. FOOTER
# =========================================================

st.html("""
<div class="footer">

    <div style="font-size:25px; font-weight:850; color:white;">
        ✦ Sentiment<span style="color:#a78bfa;">AI</span>
    </div>

    <p>
        Making movie reviews easier to understand with AI.
    </p>

    <p>
        Built with Python · TensorFlow · Keras · Streamlit
    </p>

    <p>
        Deep Learning | Natural Language Processing | Simple RNN
    </p>

    <p style="margin-top:20px;">
        © 2026 Sentiment AI. Built as a Deep Learning project.
    </p>

</div>
""")
