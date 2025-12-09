import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


st.title("Fake News Detection Application")

# Function to clean the text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

st.subheader("Upload Your Datasets")
fake_file = st.file_uploader("Upload Fake.csv", type=["csv"])
true_file = st.file_uploader("Upload True.csv", type=["csv"])

@st.cache_resource
def load_and_prepare_data(fake_df, true_df):

    fake_df["class"] = 0
    true_df["class"] = 1

    # Remove the last 10 rows (to match the notebook)
    fake_df = fake_df.iloc[:-10]
    true_df = true_df.iloc[:-10]

    df_merge = pd.concat([fake_df, true_df], axis=0)
    df_merge = df_merge.drop(["title", "subject", "date"], axis=1)
    df_merge = df_merge.sample(frac=1).reset_index(drop=True)
    df_merge["text"] = df_merge["text"].apply(wordopt)

    x = df_merge["text"]
    y = df_merge["class"]
    return x, y

@st.cache_resource
def train_models(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    LR = LogisticRegression()
    LR.fit(xv_train, y_train)

    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)

    GBC = GradientBoostingClassifier(
        random_state=0, n_estimators=50, max_depth=2, subsample=0.8, max_features='sqrt'
    )
    GBC.fit(xv_train, y_train)

    RFC = RandomForestClassifier(
        random_state=0, n_estimators=100, max_depth=10, n_jobs=-1
    )
    RFC.fit(xv_train, y_train)

    return vectorization, LR, DT, GBC, RFC, xv_test, y_test

def output_lable(n):
    return "Fake News" if n == 0 else "Not A Fake News"


# ---- MAIN LOGIC ----
if fake_file is not None and true_file is not None:
    st.success("Datasets uploaded successfully!")

    fake_df = pd.read_csv(fake_file)
    true_df = pd.read_csv(true_file)

    x, y = load_and_prepare_data(fake_df, true_df)
    vectorization, LR, DT, GBC, RFC, xv_test, y_test = train_models(x, y)

    st.subheader("Model Performance on Test Set (Accuracy):")
    st.write(f"Logistic Regression: {LR.score(xv_test, y_test):.4f}")
    st.write(f"Decision Tree: {DT.score(xv_test, y_test):.4f}")
    st.write(f"Gradient Boosting: {GBC.score(xv_test, y_test):.4f}")
    st.write(f"Random Forest: {RFC.score(xv_test, y_test):.4f}")

    st.subheader("Enter a news article to classify:")
    news_input = st.text_area("", "", height=200)

    if st.button("Classify News"):
        if news_input:
            processed_news = wordopt(news_input)
            new_xv_test = vectorization.transform([processed_news])

            pred_LR = LR.predict(new_xv_test)
            pred_DT = DT.predict(new_xv_test)
            pred_GBC = GBC.predict(new_xv_test)
            pred_RFC = RFC.predict(new_xv_test)

            st.subheader("Prediction Results:")
            st.write(f"**Logistic Regression Prediction:** {output_lable(pred_LR[0])}")
            st.write(f"**Decision Tree Prediction:** {output_lable(pred_DT[0])}")
            st.write(f"**Gradient Boosting Prediction:** {output_lable(pred_GBC[0])}")
            st.write(f"**Random Forest Prediction:** {output_lable(pred_RFC[0])}")

        else:
            st.warning("Please enter some text to classify.")

else:
    st.info("Please upload both Fake.csv and True.csv to proceed.")
