# import streamlit as st
# import pandas as pd
# from textblob import TextBlob
# import time

# # Function to clean lyrics
# def clean_lyrics(text):
#     return text.replace("\n", "")

# # Function to extract the first line of lyrics
# def extract_first_line(text):
#     return text.split("\n")[0]

# # Function to perform sentiment analysis
# def analyze_sentiment(lyric):
#     tb = TextBlob(lyric)
#     sentiment = tb.sentiment.polarity
#     subjectivity = tb.sentiment.subjectivity
#     return sentiment, subjectivity

# # Streamlit app
# def main():
#     st.title('Lyrics Sentiment Analysis')
#     st.write('Upload a CSV file with lyrics for sentiment analysis.')

#     # File upload
#     uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

#     if uploaded_file is not None:
#         # Load data from uploaded file
#         try:
#             df = pd.read_csv(uploaded_file)
#         except Exception as e:
#             st.error(f"Error: {e}")
#             return

#         st.write(f"Number of rows in the dataset: {len(df)}")

#         # Clean lyrics column
#         if 'text' in df.columns:  # Check if 'text' column exists
#             df['cleaned_lyrics'] = df['text'].apply(clean_lyrics)
#         else:
#             st.error("File does not contain 'text' column. Please upload a valid dataset.")
#             return

#         # User input for number of songs to analyze
#         num_songs = st.number_input("Enter number of songs to analyze", min_value=1, max_value=len(df), value=10)

#         if st.button('Analyze Sentiment'):
#             with st.spinner('Analyzing...'):
#                 # Extract the first line of lyrics
#                 df['first_line'] = df['cleaned_lyrics'].apply(extract_first_line)

#                 # Perform sentiment analysis on the first lines of lyrics
#                 df['sentiment'] = df['first_line'].apply(lambda x: analyze_sentiment(x)[0])
#                 df['subjectivity'] = df['first_line'].apply(lambda x: analyze_sentiment(x)[1])

#                 # Display results using st.table
#                 st.subheader("Sentiment Analysis Results")
#                 st.table(df[['artist', 'song', 'sentiment', 'subjectivity']].head(num_songs))

# if __name__ == '__main__':
#     main()

#======================================================
# Import required libraries
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

# Preprocessing Function
def preprocess_text(text_series):
    """
    Preprocess a pandas Series of text data:
    - Remove newline characters.
    - Convert to lowercase.
    - Remove stopwords.
    """
    stop_words = set(stopwords.words('english'))  # Load English stopwords
    
    def clean_text(text):
        text = text.replace("\n", " ").lower()  # Clean newlines and lowercase
        words = text.split()  # Tokenize words
        filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
        return " ".join(filtered_words)
    
    return text_series.apply(clean_text)

# Sentiment Analysis Function
def analyze_sentiment(lyrics):
    """
    Perform sentiment analysis on lyrics using TextBlob.
    Returns polarity and subjectivity.
    """
    tb = TextBlob(lyrics)
    return tb.sentiment.polarity, tb.sentiment.subjectivity

# Streamlit App
def main():
    st.title("Song Lyrics Sentiment Analysis")
    st.write("Analyze the sentiment of song lyrics by either uploading a file or manually inputting the lyrics.")

    # Sidebar Dropdown for Mode Selection
    st.sidebar.title("Options")
    mode = st.sidebar.selectbox("Choose Input Method", ["Upload File", "Manual Input"])

    # Variable to store analysis result
    analysis_results = None

    if mode == "Upload File":
        # Option to upload a CSV file
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(data.head())
            
            if 'text' in data.columns:
                if st.button("Analyze"):
                    # Preprocess and analyze
                    data['cleaned_lyrics'] = preprocess_text(data['text'])
                    sentiments = []
                    subjectivities = []
                    for lyric in data['cleaned_lyrics']:
                        sentiment, subjectivity = analyze_sentiment(lyric)
                        sentiments.append(sentiment)
                        subjectivities.append(subjectivity)
                    data['sentiment_score'] = sentiments
                    data['subjectivity'] = subjectivities

                    # Show results
                    st.write("Sentiment Analysis Results:")
                    st.dataframe(data[['text', 'cleaned_lyrics', 'sentiment_score', 'subjectivity']].head())
            else:
                st.error("The uploaded CSV must have a 'text' column containing the lyrics.")

    elif mode == "Manual Input":
        # Option to manually input lyrics
        user_lyrics = st.text_area("Enter song lyrics here")
        if user_lyrics:
            if st.button("Analyze"):
                # Preprocess and analyze
                cleaned_lyrics = preprocess_text(pd.Series([user_lyrics])).iloc[0]
                sentiment, subjectivity = analyze_sentiment(cleaned_lyrics)

                # Display results
                st.write("**Original Lyrics:**")
                st.text(user_lyrics)
                st.write("**Cleaned Lyrics:**")
                st.text(cleaned_lyrics)
                st.write("**Sentiment Analysis Results:**")
                st.write(f"Polarity: {sentiment}")
                st.write(f"Subjectivity: {subjectivity}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
