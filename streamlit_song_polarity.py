import streamlit as st
import pandas as pd
from textblob import TextBlob

# Function to clean lyrics
def clean_lyrics(text):
    return text.replace("\n", "")

# Function to extract the first line of lyrics
def extract_first_line(text):
    return text.split("\n")[0]

# Function to perform sentiment analysis
def analyze_sentiment(lyric):
    tb = TextBlob(lyric)
    sentiment = tb.sentiment.polarity
    subjectivity = tb.sentiment.subjectivity
    return sentiment, subjectivity

# Streamlit app
def main():
    st.title('Lyrics Sentiment Analysis')
    st.write('Upload a CSV file with lyrics for sentiment analysis.')

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

    if uploaded_file is not None:
        # Load data from uploaded file
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error: {e}")
            return

        st.write(f"Number of rows in the dataset: {len(df)}")

        # Clean lyrics column
        if 'text' in df.columns:  # Check if 'text' column exists
            df['cleaned_lyrics'] = df['text'].apply(clean_lyrics)
        else:
            st.error("File does not contain 'text' column. Please upload a valid dataset.")
            return

        # Extract the first line of lyrics
        df['first_line'] = df['cleaned_lyrics'].apply(extract_first_line)

        # Perform sentiment analysis on the first line of lyrics
        df['sentiment'] = df['first_line'].apply(lambda x: analyze_sentiment(x)[0])
        df['subjectivity'] = df['first_line'].apply(lambda x: analyze_sentiment(x)[1])

        # Display results
        st.write(df[['artist', 'song', 'sentiment', 'subjectivity']].head(10))

if __name__ == '__main__':
    main()
