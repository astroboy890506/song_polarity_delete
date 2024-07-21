import streamlit as st
import pandas as pd
from textblob import TextBlob

# Function to clean lyrics
def clean_lyrics(text):
    return text.replace("\n", "")

# Function to perform sentiment analysis
def analyze_sentiment(lyric):
    tb = TextBlob(lyric)
    sentiment = tb.sentiment.polarity
    subjectivity = tb.sentiment.subjectivity
    return sentiment, subjectivity

# Streamlit app
def main():
    st.title('Lyrics Sentiment Analysis')
    st.write('Upload an Excel file with a "text" column containing song lyrics.')

    # File upload
    uploaded_file = st.file_uploader("Upload an Excel file", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        # Load data from uploaded file
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error: {e}")
            return

        st.write(f"Number of rows in the dataset: {len(df)}")

        # Clean lyrics column
        df['cleaned_lyrics'] = df['text'].apply(clean_lyrics)

        # Select number of rows to analyze
        rows_to_analyze = st.number_input("Select number of rows to analyze", min_value=1, max_value=len(df), value=10)

        # Perform sentiment analysis
        sentiments = []
        subjectivities = []
        for i in range(rows_to_analyze):
            lyric = df.loc[i, 'cleaned_lyrics']
            sentiment, subjectivity = analyze_sentiment(lyric)
            sentiments.append(sentiment)
            subjectivities.append(subjectivity)

        # Add sentiment scores to dataframe
        df['sentiment_score'] = sentiments
        df['subjectivity'] = subjectivities

        # Display results
        st.write(df[['No', 'Song', 'Artist', 'sentiment_score', 'subjectivity']].head(rows_to_analyze))

if __name__ == '__main__':
    main()
