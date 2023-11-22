import streamlit as st
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from googletrans import Translator

# Load the dataset from the Google Drive link
url = 'https://drive.google.com/uc?id=1ZjYvMpocm1Y_PM4daTP__2HG03wqzCNl'
data = pd.read_csv(url)

# Data Preprocessing
X = data["Text"]
y = data["Language"]

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y)

# Text preprocessing
data_list = []
for text in X:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    data_list.append(text)

# Hashing Vectorizer
vectorizer = HashingVectorizer(n_features=2**10, alternate_sign=False)
X = vectorizer.transform(data_list)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Model creation and prediction
model = MultinomialNB()
model.fit(x_train, y_train)

# Streamlit app
st.set_page_config(
    page_title="Language Detection App",
    page_icon=":globe_with_meridians:",
    layout="wide",
)

# Header
st.title("Language Detection App")
st.write("Enter text in the sidebar to detect its language.")

# Sidebar for language detection
text_input = st.sidebar.text_area("Enter text:", height=300, max_chars=None)  # Setting max_chars to None for unlimited input

target_language = st.sidebar.text_input("Enter target language for translation (e.g., 'German', 'French'):")

prediction = ""  # Initialize prediction with an empty string

if st.sidebar.button("Detect Language", key="detect_language"):
    text_input = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text_input)
    text_input = re.sub(r'[[]]', ' ', text_input)
    text_input = text_input.lower()

    user_input = vectorizer.transform([text_input])
    prediction = le.inverse_transform(model.predict(user_input))[0]

    st.sidebar.subheader("Language Prediction:")
    st.sidebar.write(prediction)

# Mobile responsiveness
if st.sidebar.checkbox("Show instructions", False):
    st.sidebar.markdown(
        """
        **Instructions:**
        - Enter the text you want to detect in the sidebar.
        - Click the "Detect Language" button to see the prediction.
        - Enter the target language for translation.
        """
    )

# Footer
st.sidebar.text("Developed by Jawad Ahmad")

# Optional: Add a translation button
if prediction and target_language:
    translator = Translator()
    translated_text = translator.translate(text_input, src=prediction, dest=target_language).text
    st.subheader(f"Translated Text (to {target_language}):")
    st.write(translated_text)
