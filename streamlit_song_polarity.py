import streamlit as st
import pandas as pd
from textblob import TextBlob
import io

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
    st.write('Upload a file with lyrics for sentiment analysis.')
    st.write('Supported file formats: Excel (xlsx, xls) and CSV (csv)')

    # File upload
    uploaded_file = st.file_uploader("Upload a file", type=['xlsx', 'xls', 'csv'])

    if uploaded_file is not None:
        # Determine file type
        file_ext = uploaded_file.name.split('.')[-1]
        
        try:
            if file_ext in ['xlsx', 'xls']:
                # Load Excel file
                df = pd.read_excel(uploaded_file)
            elif file_ext == 'csv':
                # Load CSV file
                df = pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload an Excel (xlsx, xls) or CSV (csv) file.")
                return
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
        st.write(f"Length of sentiments list: {len(sentiments)}")
        st.write(f"Length of subjectivities list: {len(subjectivities)}")
        
        df['sentiment_score'] = sentiments
        df['subjectivity'] = subjectivities

        # Display results
        st.write(df[['No', 'Song', 'Artist', 'sentiment_score', 'subjectivity']].head(rows_to_analyze))

if __name__ == '__main__':
    main()
