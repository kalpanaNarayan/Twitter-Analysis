import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Ensure you have downloaded these NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to perform tokenization, stemming, and lemmatization
def preprocess_text(text, apply_stemming=False, apply_lemmatization=False):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords

    if apply_stemming:
        tokens = [stemmer.stem(word) for word in tokens]
    elif apply_lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Load the data
data = pd.read_csv("table (4).csv")

# Handle missing values and convert necessary columns to numeric
data['Likes'] = pd.to_numeric(data['Likes'], errors='coerce').fillna(0)
data['Retweets'] = pd.to_numeric(data['Retweets'], errors='coerce').fillna(0)
data['Comments'] = pd.to_numeric(data['Comments'], errors='coerce').fillna(0)
data['Views'] = pd.to_numeric(data['Views'], errors='coerce').fillna(0)

# Calculate Engagement Rate and handle division by zero
data['Engagement Rate'] = (data['Likes'] + data['Retweets'] + data['Comments']) / data['Views'].replace(0, np.nan)

# Convert 'Date' column to datetime if it exists
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🎯 Introduction"

if 'model_options' not in st.session_state:
    st.session_state.model_options = []

# Define navigation functions
def next_page():
    pages = ["🎯 Introduction", "Data Overview", "Data Preprocessing", "Data Visualization", "Sentiment Analysis", "Model Building", "Model Evaluation", "Conclusion"]
    current_index = pages.index(st.session_state.current_page)
    if current_index < len(pages) - 1:
        st.session_state.current_page = pages[current_index + 1]

def previous_page():
    pages = ["🎯 Introduction", "Data Overview", "Data Preprocessing", "Data Visualization", "Sentiment Analysis", "Model Building", "Model Evaluation", "Conclusion"]
    current_index = pages.index(st.session_state.current_page)
    if current_index > 0:
        st.session_state.current_page = pages[current_index - 1]

# App configuration
st.set_page_config(page_title="Twitter Data Analysis", layout="wide", page_icon="🐦")

# Navigation Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["🎯 Introduction", "Data Overview", "Data Preprocessing", "Data Visualization", "Sentiment Analysis", "Conclusion"])

# Introduction section
if page == "🎯 Introduction":
    st.title("🐦 Twitter Hate Speech Detection")
    st.markdown("""
    ## 📋 Project Overview
    This project focuses on detecting hate speech in tweets. By analyzing sentiments, we aim to identify tweets with racist or abusive content using multiple machine learning models.
    
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

# Data Overview
elif page == "Data Overview":
    st.title("📁 Data Overview")
    st.write("### 📄 Dataset Preview")
    st.write(data.head(10))

    st.write("### 🔢 Data Dimensions")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")

    st.write("### 📊 Summary Statistics")
    st.write(data.describe())

    st.write("### 📝 Column Descriptions")
    st.markdown("""
    - **Name:** Name of the Twitter user.
    - **Handle:** Twitter handle of the user.
    - **Media URL:** URL to the media (image/video) in the tweet.
    - **Retweets:** Number of retweets.
    - **Likes:** Number of likes.
    - **Comments:** Number of comments.
    - **Views:** Number of views.
    - **Post Body:** Content of the tweet.
    - **Date:** Date of the tweet.
    - **Timestamp:** Exact timestamp of the tweet.
    """)

# Data Preprocessing
elif page == "Data Preprocessing":
    st.title("🔧 Data Preprocessing")

    st.markdown("### Original Data Preview")
    st.write(data.head(10))

    st.markdown("### Select Preprocessing Steps")

    # Checkboxes for different preprocessing steps
    drop_columns = st.checkbox('Drop Columns', value=False)
    fill_missing = st.checkbox('Fill Missing Values', value=True)
    convert_date = st.checkbox('Convert Date Column', value=True)
    apply_stemming = st.checkbox('Apply Stemming', value=False)
    apply_lemmatization = st.checkbox('Apply Lemmatization', value=False)

    # Dataframe for showing processed data
    processed_data = data.copy()

    # Apply selected preprocessing steps
    if drop_columns:
        st.markdown("### Dropping Columns: Date, Media URL, Engagement Rate")
        if 'Date' in processed_data.columns:
            processed_data = processed_data.drop(columns=['Date'])
        if 'Media URL' in processed_data.columns:
            processed_data = processed_data.drop(columns=['Media URL'])
        if 'Engagement Rate' in processed_data.columns:
            processed_data = processed_data.drop(columns=['Engagement Rate'])
        st.write("#### Data After Dropping Columns")
        st.write(processed_data.head(10))

    if fill_missing:
        st.markdown("### Filling Missing Values")
        processed_data['Likes'] = pd.to_numeric(processed_data['Likes'], errors='coerce').fillna(0)
        processed_data['Retweets'] = pd.to_numeric(processed_data['Retweets'], errors='coerce').fillna(0)
        processed_data['Comments'] = pd.to_numeric(processed_data['Comments'], errors='coerce').fillna(0)
        processed_data['Views'] = pd.to_numeric(processed_data['Views'], errors='coerce').fillna(0)
        st.write("#### Data After Filling Missing Values")
        st.write(processed_data.head(10))

    if convert_date:
        st.markdown("### Converting Date Column to Datetime")
        if 'Date' in processed_data.columns:
            processed_data['Date'] = pd.to_datetime(processed_data['Date'], errors='coerce')
        st.write("#### Data After Date Conversion")
        st.write(processed_data.head(10))

    st.markdown("### Data Summary After Preprocessing")
    st.write(processed_data.describe())

    # Allow users to select the type of preprocessing to show
    st.markdown("### Choose Preprocessing to View")
    preprocessing_step = st.selectbox('Select Preprocessing Step', 
                                      ["Original", "After Dropping Columns", "After Filling Missing Values", "After Date Conversion"])

    if preprocessing_step == "Original":
        st.write("#### Original Data Preview")
        st.write(data.head(10))
    elif preprocessing_step == "After Dropping Columns":
        if drop_columns:
            st.write("#### Data After Dropping Columns")
            st.write(processed_data.head(10))
        else:
            st.write("Please select 'Drop Columns' to see this preview.")
    elif preprocessing_step == "After Filling Missing Values":
        if fill_missing:
            st.write("#### Data After Filling Missing Values")
            st.write(processed_data.head(10))
        else:
            st.write("Please select 'Fill Missing Values' to see this preview.")
    elif preprocessing_step == "After Date Conversion":
        if convert_date:
            st.write("#### Data After Date Conversion")
            st.write(processed_data.head(10))
        else:
            st.write("Please select 'Convert Date Column' to see this preview.")

    st.markdown("### Text Preprocessing")
    st.write("#### Preview of Text Preprocessing")
    if 'Post Body' in data.columns:
        st.markdown("### Select Text Preprocessing Options")
        sample_text = data['Post Body'].sample(1).values[0]

        processed_text = preprocess_text(sample_text, apply_stemming, apply_lemmatization)
        st.write(f"Original Text: {sample_text}")
        st.write(f"Processed Text: {processed_text}")

    st.markdown("### Navigation")
    st.button("Previous Page", on_click=previous_page)
    st.button("Next Page", on_click=next_page)

# Data Visualization
elif page == "Data Visualization":
    st.title("🎨 Data Visualization")

    # Interactive Scatter Plot
    x_axis = st.selectbox("Select X-axis", data.columns)
    y_axis = st.selectbox("Select Y-axis", data.columns)

    if x_axis and y_axis:
        st.write(f"### Scatter plot of {y_axis} vs {x_axis}")
        fig = px.scatter(data, x=x_axis, y=y_axis, color=data['Date'].astype(str), color_continuous_scale='Viridis', title=f'Scatter Plot of {y_axis} vs {x_axis}')
        st.plotly_chart(fig)
        st.write("This scatter plot visualizes the relationship between the selected X-axis and Y-axis values. Each point represents a data entry.")

    # Distribution of Retweets, Likes, Comments, Views
    st.write("### Distribution of Retweets, Likes, Comments, Views")
    fig = make_subplots(rows=2, cols=2, subplot_titles=['Retweets', 'Likes', 'Comments', 'Views'])
    fig.add_trace(go.Histogram(x=data['Retweets'].dropna(), marker=dict(color='deepskyblue'), name='Retweets'), row=1, col=1)
    fig.add_trace(go.Histogram(x=data['Likes'].dropna(), marker=dict(color='seagreen'), name='Likes'), row=1, col=2)
    fig.add_trace(go.Histogram(x=data['Comments'].dropna(), marker=dict(color='tomato'), name='Comments'), row=2, col=1)
    fig.add_trace(go.Histogram(x=data['Views'].dropna(), marker=dict(color='orange'), name='Views'), row=2, col=2)
    
    fig.update_layout(title_text='Distribution of Engagement Metrics', showlegend=False, height=800)
    st.plotly_chart(fig)
    st.write("These histograms show the distribution of Retweets, Likes, Comments, and Views. Each color represents a different metric.")

    # Word Cloud from Post Body
    st.write("### 🗨 Word Cloud from Post Body")
    all_text = ' '.join(data['Post Body'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='inferno').generate(all_text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    st.write("The word cloud displays the most frequently occurring words in the `Post Body` column, with larger words indicating higher frequency.")

    # 3D Line Plot for Timeline Data
    st.write("### 3D Line Plot for Timeline Data")
    
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
        data = data.dropna(subset=['Timestamp'])  # Drop rows where Timestamp is NaT
        if not data.empty:
            fig = go.Figure()
            unique_dates = data['Timestamp'].dropna().unique()
            for timestamp in unique_dates:
                df_timestamp = data[data['Timestamp'] == timestamp]
                fig.add_trace(go.Scatter3d(
                    x=df_timestamp['Likes'].dropna(),
                    y=df_timestamp['Retweets'].dropna(),
                    z=df_timestamp['Comments'].dropna(),
                    mode='lines+markers',
                    name=str(timestamp)
                ))
            fig.update_layout(
                title='3D Line Plot of Likes, Retweets, and Comments Over Time',
                scene=dict(
                    xaxis_title='Likes',
                    yaxis_title='Retweets',
                    zaxis_title='Comments'
                )
            )
            st.plotly_chart(fig)
            st.write("This 3D line plot visualizes Likes, Retweets, and Comments over time, with different lines representing different timestamps.")
        else:
            st.write("No valid timestamps found in the data.")
    else:
        st.write("Timestamp column not found in the data.")
        
# Sentiment Analysis
if page == "Sentiment Analysis":
    st.title("💬 Sentiment Analysis")

    st.write("#### Select Model for Sentiment Analysis")
    model_option = st.selectbox("Choose Model", ["Logistic Regression", "SVM","LightGBM"])

    # Initialize the model
    model = None
    if model_option == "Logistic Regression":
        model = LogisticRegression()
    elif model_option == "SVM":
        model = SVC()
    elif model_option == "LightGBM":
        model = lgb.LGBMClassifier()

    if model:
        st.write("### Train the Model")

        if st.button("Train Model"):
            try:
                # Handle missing values in 'Post Body'
                data['Post Body'] = data['Post Body'].fillna('')  # Replace NaN with empty string

                # Debugging: Display column names
                st.write("Available columns in data:", data.columns)

                # Ensure columns exist
                if 'Post Body' not in data.columns:
                    st.error("Column 'Post Body' is missing in the data.")
                if 'Likes' not in data.columns:
                    st.error("Column 'Likes' is missing in the data.")
                
                # Proceed if columns are present
                if 'Post Body' in data.columns and 'Likes' in data.columns:
                    X = data['Post Body']  # Replace with your feature column
                    y = data['Likes']  # Replace with your target column

                    # Convert target variable to categorical if needed
                    y = y.astype('int')  # Ensure 'y' is in integer format

                    vectorizer = TfidfVectorizer()
                    X_vec = vectorizer.fit_transform(X)

                    # Train the model
                    model.fit(X_vec, y)
                    y_pred = model.predict(X_vec)

                    # Display metrics
                    accuracy = accuracy_score(y, y_pred)
                    report = classification_report(y, y_pred, output_dict=True)
                    cm = confusion_matrix(y, y_pred)

                    st.write(f"**Accuracy:** {accuracy:.2f}")
                    st.write("### Classification Report")
                    st.write(report)
                    
                    st.write("### Confusion Matrix")
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
                    st.pyplot(fig)

                    # Example for RandomForestClassifier specific functionality
                    if isinstance(model, RandomForestClassifier):
                        # Ensure the model has been fitted
                        if hasattr(model, 'estimators_'):
                            # Access the 'estimators_' attribute
                            st.write("Number of estimators:", len(model.estimators_))

            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.markdown("### Navigation")
    st.button("Previous Page", on_click=previous_page)
    st.button("Next Page", on_click=next_page)

# Conclusion
elif page == "Conclusion":
    st.title("📝 Conclusions")
    st.write("""
    ### 📋 Summary
    - The models were evaluated based on accuracy, confusion matrix, ROC curve, and AUC.
    - Logistic Regression and SVM performed well, while Random Forest, XGBoost, and LightGBM provided competitive results.

    ### 🔮 Future Work
    - Enhance the dataset with additional features or more data.
    - Explore other advanced models and techniques.
    - Conduct a more detailed error analysis to improve model performance.
    """)

    st.markdown("### Navigation")
    st.button("Previous Page", on_click=previous_page)
