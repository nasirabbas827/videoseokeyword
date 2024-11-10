import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import nltk
import re
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load dataset
file_path = 'NewDataset.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Rename columns as per requirement
df = df.rename(columns={
    'views_count': 'views_count',
    'like_count': 'likes_count',
    'comment_count': 'comments_count'
})

# Step 1: Create or label 'Engagement Label' column
def create_engagement_label(row):
    # Example thresholds for "Good", "Average", and "Bad" videos
    if row['views_count'] > 10000 and row['likes_count'] > 5000:
        return 'Good'
    elif row['views_count'] > 5000 and row['likes_count'] > 200:
        return 'Average'
    else:
        return 'Bad'

df['Engagement Label'] = df.apply(create_engagement_label, axis=1)

# Preprocessing Data: Clean the dataset
def preprocess_data(df):
    # Drop nulls and duplicates
    df = df.dropna(subset=['Title', 'Description'])
    df = df.drop_duplicates(subset=['Title', 'Description'])
    
    # Fill NaN in columns
    df['Title'] = df['Title'].fillna('')
    df['Description'] = df['Description'].fillna('')
    
    # Clean the Title and Description (tokenization, stopword removal, etc.)
    df['Cleaned Title'] = df['Title'].apply(lambda x: clean_text(x))
    df['Cleaned Description'] = df['Description'].apply(lambda x: clean_text(x))
    
    return df

# Text cleaning function
def clean_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)

    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize the words
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    
    # Return cleaned text
    return ' '.join(tokens)

# Preprocess the data
df = preprocess_data(df)

# Function to analyze keywords using NLTK
def analyze_keywords(data):
    # Combine all titles, descriptions, and tags into one text
    combined_text = ' '.join(data['Cleaned Title'].tolist() + data['Cleaned Description'].tolist() + data['Tags'].astype(str).tolist())
    
    # Tokenize the combined text
    words = word_tokenize(combined_text.lower())
    
    # Filter out stop words and non-alphabetic words
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Create frequency distribution of the keywords
    freq_dist = FreqDist(filtered_words)
    
    keywords_df = pd.DataFrame(freq_dist.items(), columns=['keyword', 'frequency'])
    
    return keywords_df.sort_values(by='frequency', ascending=False)

# Function to calculate TF-IDF and cosine similarity
def calculate_cosine_similarity(df, query):
    # Create TF-IDF vectors from the 'Cleaned Title' and 'Cleaned Description' columns
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Cleaned Title'] + " " + df['Cleaned Description'])
    
    # Transform the user's query
    query_tfidf = vectorizer.transform([query])
    
    # Calculate cosine similarity between query and videos
    sim_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    return sim_scores, vectorizer

# Modify the suggest_video_titles function
def suggest_video_titles(query, num_recommendations=5, top_keywords=10):
    sim_scores, vectorizer = calculate_cosine_similarity(df, query)
    
    # Get the indices of the most similar videos
    sim_scores_indices = np.argsort(sim_scores)[-num_recommendations-1:-1][::-1]
    
    # Prepare the top recommended titles and related keywords
    recommendations = df.iloc[sim_scores_indices][['Title', 'Description', 'views_count', 'likes_count']]
    recommendations['similarity_score'] = sim_scores[sim_scores_indices]
    
    # Define engagement prediction based on similarity score
    def get_engagement_prediction(score):
        if score > 0.40:
            return 'Good'
        elif score > 0.20:
            return 'Average'
        else:
            return 'Bad'
    
    recommendations['engagement_prediction'] = recommendations['similarity_score'].apply(get_engagement_prediction)
    
    # Extract top N keywords for each recommended video using TF-IDF
    recommendations['related_keywords'] = recommendations.apply(lambda row: extract_keywords_from_text(row['Title'], row['Description'], vectorizer), axis=1)
    
    return recommendations

# Function to extract top N relevant keywords for a given text (title + description)
def extract_keywords_from_text(title, description, vectorizer, top_n=10):
    # Combine title and description into one text
    text = title + " " + description
    
    # Transform the text into TF-IDF vector
    tfidf_matrix = vectorizer.transform([text])
    
    # Get feature names (words)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Get TF-IDF scores for the text
    tfidf_scores = tfidf_matrix.toarray().flatten()
    
    # Sort by scores and get the top N keywords
    top_indices = np.argsort(tfidf_scores)[-top_n:][::-1]
    
    # Get the top N keywords
    top_keywords = feature_names[top_indices]
    
    return ", ".join(top_keywords)

# Function to extract top 20 relevant keywords
def extract_relevant_keywords(df, num_keywords=20):
    keywords_analysis = analyze_keywords(df)
    
    # Filter out generic words that are not relevant to the video content
    generic_keywords = ['video', 'channel', 'course', 'subscribe', 'follow', 'instagram', 'youtube', 'facebook', 'tiktok', 'whatsapp', 'learn']
    keywords_analysis = keywords_analysis[~keywords_analysis['keyword'].isin(generic_keywords)]
    
    # Return top N relevant keywords
    return keywords_analysis.head(num_keywords)

# Define the machine learning model and its pipeline
def train_model(X, y):
    # Create a pipeline with TF-IDF vectorizer and SVM classifier
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                         ('scaler', StandardScaler(with_mean=False)),
                         ('svm', SVC())])  # Support Vector Machine for classification

    # Train the model
    pipeline.fit(X, y)

    # Predict on the training set (or you could use the test set here)
    y_pred = pipeline.predict(X)

    # Calculate accuracy, recall, and F1 score
    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred, average='weighted')  # Use 'weighted' for multi-class
    f1 = f1_score(y, y_pred, average='weighted')  # Use 'weighted' for multi-class

    # Print the metrics to the console
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return pipeline

# Train the model using cleaned data (combined text: Title + Description)
X_train = df[['Cleaned Title', 'Cleaned Description']]
X_train['Combined Text'] = X_train['Cleaned Title'] + " " + X_train['Cleaned Description']
y_train = df['Engagement Label']  # Engagement label ("Good", "Bad", "Average")

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_train['Combined Text'], y_train, test_size=0.2, random_state=42)

# Train the model
model = train_model(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    
    # Get recommended video titles and relevant keywords
    recommended_videos = suggest_video_titles(query)
    relevant_keywords = extract_relevant_keywords(df)
    
    # Predict the engagement label (Good, Bad, Average) for the user-provided query
    predicted_engagement = model.predict([query])
    
    # Calculate model evaluation metrics on the test set (after training)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print metrics to the console
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return render_template('results.html', 
                           recommendations=recommended_videos, 
                           keywords=relevant_keywords, 
                           predicted_engagement=predicted_engagement[0],
                           search_query=query,
                           accuracy=accuracy, 
                           recall=recall, 
                           f1=f1)  # Pass the metrics to the template

if __name__ == '__main__':
    app.run(debug=True)
