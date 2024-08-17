import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Load stopwords
stop_words = set(ENGLISH_STOP_WORDS)

# Define preprocessing functions
def preprocess_text(text, lowercase=True, remove_special=True, remove_stopwords=True, stemming=False):
    if lowercase:
        text = text.lower()
    if remove_special:
        text = re.sub(r'[^a-zA-Z\s]', '', text)
    if remove_stopwords:
        words = text.split()
        words = [word for word in words if word not in stop_words]
        text = ' '.join(words)
    if stemming:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        words = text.split()
        text = ' '.join([ps.stem(word) for word in words])
    return text

def preprocess_data(df, text_column='tweet', **kwargs):
    if text_column in df.columns:
        df['cleaned_text'] = df[text_column].apply(lambda x: preprocess_text(x, **kwargs))
    else:
        raise KeyError(f"Column '{text_column}' is missing in the dataframe.")
    return df

# App configuration with title and layout
st.set_page_config(
    page_title="Twitter Hate Speech Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🐦"
)

# Adding custom CSS for background and text styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    h1, h2, h3, h4 {
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load datasets
train = pd.read_csv("Twitter Sentiments.csv")
test = pd.read_csv("test_tweets_anuFYb8.csv")

# Data Preprocessing
st.sidebar.title("🔍 Navigation")
page = st.sidebar.selectbox("Choose a section", [
    "🎯 Introduction", 
    "🗂 Data Overview", 
    "🛠 Data Preprocessing", 
    "📊 Model Building", 
    "🔍 Model Evaluation", 
    "📈 Performance Metrics", 
    "🎨 Visualizations", 
    "📝 Conclusions"
])

# Introduction section
if page == "🎯 Introduction":
    st.title("🐦 Twitter Hate Speech Detection")
    st.markdown("""
    ## 📋 Project Overview
    This project focuses on detecting hate speech in tweets. By analyzing sentiments, we aim to identify tweets with racist or sexist content using multiple machine learning models.
    
    *Assignment by Kalpana N | ESE-1 | 2347229*
    """)
    st.image("Twitter-rebrands-X.webp", use_column_width=True)
    st.markdown("""
    ### 🔍 Problem Statement
    The task is to develop a model that can classify tweets as hate speech or not. This is essential for creating a safer online environment.
    
    ### 📚 Objectives
    - Understand the data distribution and characteristics.
    - Preprocess the data for model building.
    - Train various machine learning models.
    - Evaluate model performance and visualize the results.
    - Provide insights and conclusions based on the analysis.
    """)

# Data Overview section
elif page == "🗂 Data Overview":
    st.title("📁 Data Overview")
    st.write("### 📄 Training Dataset")
    st.write(train.head(10))

    st.write("### 📄 Test Dataset")
    st.write(test.head(10))

    st.write("### 🔢 Data Dimensions")
    st.write(f"Training set: {train.shape}")
    st.write(f"Test set: {test.shape}")

    st.write("### 📊 Target Distribution in Training Data")
    fig, ax = plt.subplots()
    sns.countplot(x='label', data=train, ax=ax)
    st.pyplot(fig)

    st.markdown("""
    ### ✏ Observations
    - The dataset is imbalanced with more non-hate speech tweets compared to hate speech tweets.
    - This imbalance needs to be considered when building and evaluating models.
    """)

# Data Preprocessing section
elif page == "🛠 Data Preprocessing":
    st.title("🔧 Data Preprocessing")
    st.markdown("Select the preprocessing steps you want to apply to the text data:")

    # Preprocessing options
    lowercase = st.checkbox("Convert to lowercase", value=True)
    remove_special = st.checkbox("Remove special characters and numbers")
    remove_stopwords = st.checkbox("Remove stopwords")
    stemming = st.checkbox("Apply stemming")
    
    # Apply preprocessing
    st.markdown("### Preprocessing the Training Dataset")
    try:
        train = preprocess_data(train, text_column='tweet', lowercase=lowercase, remove_special=remove_special, remove_stopwords=remove_stopwords, stemming=stemming)
        st.write(train.head(10))
    except KeyError as e:
        st.write(f"Error: {e}")

    st.markdown("### Preprocessing the Test Dataset")
    try:
        test = preprocess_data(test, text_column='tweet', lowercase=lowercase, remove_special=remove_special, remove_stopwords=remove_stopwords, stemming=stemming)
        st.write(test.head(10))
    except KeyError as e:
        st.write(f"Error: {e}")

    # Add a button to trigger model building
    if st.button('Proceed to Model Building'):
        if 'cleaned_text' in train.columns and 'cleaned_text' in test.columns:
            st.session_state.preprocessing_done = True
            st.success("Preprocessing completed! Navigating to the next section...")
            st.experimental_rerun()  # Automatically navigate to model building page
        else:
            st.session_state.preprocessing_done = False
            st.write("Preprocessing is incomplete. Ensure that 'cleaned_text' column exists in both datasets.")

# Initialize vectorizer and model placeholders
vectorizer = TfidfVectorizer(max_features=5000)
models = {}

# Model Building section
elif page == "📊 Model Building":
    st.title("🤖 Model Building")
    
    if st.session_state.get('preprocessing_done', False):
        st.markdown("""
        ### Model Options
        Choose a model to train:
        - *Logistic Regression*: Simple and interpretable.
        - *Support Vector Machine (SVM)*: Effective in high-dimensional spaces.
        - *Random Forest*: Ensemble method that improves accuracy.
        - *XGBoost*: Efficient and scalable gradient boosting.
        - *LightGBM*: Fast and effective for large datasets.
        """)
        
        model_options = st.multiselect("Select models to build", [
            "Logistic Regression", 
            "Support Vector Machine", 
            "Random Forest", 
            "XGBoost", 
            "LightGBM"
        ])
        
        # Vectorize and train selected models
        X_train = vectorizer.fit_transform(train['cleaned_text'])
        y_train = train['label']
        
        if "Logistic Regression" in model_options:
            st.markdown("### 🛠 Logistic Regression Model")
            model_lr = LogisticRegression()
            model_lr.fit(X_train, y_train)
            models["Logistic Regression"] = model_lr
            st.write("Logistic Regression model trained.")

        if "Support Vector Machine" in model_options:
            st.markdown("### 🛠 Support Vector Machine (SVM) Model")
            model_svm = SVC(probability=True)
            model_svm.fit(X_train, y_train)
            models["Support Vector Machine"] = model_svm
            st.write("Support Vector Machine model trained.")

        if "Random Forest" in model_options:
            st.markdown("### 🛠 Random Forest Model")
            model_rf = RandomForestClassifier()
            model_rf.fit(X_train, y_train)
            models["Random Forest"] = model_rf
            st.write("Random Forest model trained.")

        if "XGBoost" in model_options:
            st.markdown("### 🛠 XGBoost Model")
            model_xgb = xgb.XGBClassifier()
            model_xgb.fit(X_train, y_train)
            models["XGBoost"] = model_xgb
            st.write("XGBoost model trained.")

        if "LightGBM" in model_options:
            st.markdown("### 🛠 LightGBM Model")
            model_lgb = lgb.LGBMClassifier()
            model_lgb.fit(X_train, y_train)
            models["LightGBM"] = model_lgb
            st.write("LightGBM model trained.")

        if models:
            st.session_state.models_trained = True
            st.success("Models trained successfully!")
        else:
            st.session_state.models_trained = False
            st.write("No models were selected or trained.")
    else:
        st.write("Preprocessing is incomplete. Please preprocess the data first.")

# Model Evaluation section
elif page == "🔍 Model Evaluation":
    st.title("🔍 Model Evaluation")

    if st.session_state.get('models_trained', False):
        st.write("### Model Evaluation Metrics")
        selected_model_name = st.selectbox("Select a model for evaluation", list(models.keys()))
        selected_model = models[selected_model_name]
        
        X_test = vectorizer.transform(test['cleaned_text'])
        y_test = test['label']
        
        y_pred = selected_model.predict(X_test)
        y_prob = selected_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### 📈 Accuracy: {accuracy:.2f}")
        
        st.write("### 📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=selected_model.classes_)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        st.pyplot(fig)

        st.write("### 📉 ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)
    else:
        st.write("No models have been trained yet. Please go back to the Model Building section.")

# Performance Metrics section
elif page == "📈 Performance Metrics":
    st.title("📈 Performance Metrics")
    st.markdown("""
    - **Accuracy**: Measures the proportion of correctly classified instances out of all instances.
    - **Confusion Matrix**: Provides a summary of prediction results by showing the number of true positives, true negatives, false positives, and false negatives.
    - **ROC Curve**: A graphical representation of a model's diagnostic ability, plotting the true positive rate against the false positive rate.
    - **AUC**: Area Under the ROC Curve, which quantifies the overall ability of the model to discriminate between positive and negative classes.
    """)
    st.write("### 📜 Insights")
    st.write("""
    - **Accuracy**: High accuracy indicates that the model performs well overall.
    - **Confusion Matrix**: Helps in understanding the types of errors made by the model.
    - **ROC Curve and AUC**: Useful for comparing models and understanding their performance in different classification scenarios.
    """)

# Visualizations section
elif page == "🎨 Visualizations":
    st.title("🎨 Visualizations")
    st.markdown("""
    - **Word Cloud**: Visualizes the most frequent words in the dataset.
    - **Distribution of Tweet Lengths**: Shows the length of tweets to understand their distribution.
    """)
    
    st.write("### 🗨 Word Cloud")
    all_text = ' '.join(train['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    st.write("### 📏 Distribution of Tweet Lengths")
    train['tweet_length'] = train['cleaned_text'].apply(len)
    fig, ax = plt.subplots()
    sns.histplot(train['tweet_length'], kde=True, ax=ax)
    ax.set_title('Distribution of Tweet Lengths')
    st.pyplot(fig)

# Conclusions section
elif page == "📝 Conclusions":
    st.title("📝 Conclusions")
    st.markdown("""
    - **Summary**: Provides a high-level overview of findings, model performances, and insights gained from the analysis.
    - **Future Work**: Suggests improvements or further investigations that could be conducted.
    """)
    st.write("""
    ### 📋 Summary
    - The models were evaluated based on accuracy, confusion matrix, ROC curve, and AUC.
    - Logistic Regression and SVM performed well, while Random Forest, XGBoost, and LightGBM provided competitive results.

    ### 🔮 Future Work
    - Enhance the dataset with additional features or more data.
    - Explore other advanced models and techniques.
    - Conduct a more detailed error analysis to improve model performance.
    """)
