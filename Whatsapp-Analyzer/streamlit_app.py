import streamlit as st
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_data()

# --- Helper Functions ---

def parse_whatsapp_chat(file_content):
    """Parse WhatsApp chat text content"""
    chat_data = []
    current_message = {"timestamp": None, "sender": None, "message": ""}
    
    # WhatsApp chat pattern: [DD/MM/YY, HH:MM] Sender: Message or M/D/YY, H:MM AM/PM - Sender: Message
    message_pattern = re.compile(r'^(?P<timestamp>\d{1,2}[\/\.]\d{1,2}[\/\.]\d{2,4}[,\s]+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?) [-:\s]+ (?P<sender>[^:]+): (?P<message>.*)')
    
    lines = file_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip system messages
        if any(msg in line for msg in ["Messages to this group are now end-to-end encrypted", 
                                        "You created group", "changed the subject from", 
                                        "added", "left", "deleted", "was added"]):
            continue
        
        match = message_pattern.match(line)
        if match:
            if current_message["timestamp"] is not None:
                chat_data.append(current_message)
            
            timestamp_str = match.group("timestamp").strip()
            sender = match.group("sender").strip()
            message = match.group("message").strip()
            
            # Parse timestamp
            parsed_timestamp = pd.NaT
            timestamp_formats = [
                '%m/%d/%y, %H:%M',
                '%m/%d/%Y, %H:%M',
                '%m/%d/%y, %I:%M %p',
                '%m/%d/%Y, %I:%M %p',
                '%d/%m/%y, %H:%M',
                '%d/%m/%Y, %H:%M',
                '%d/%m/%y, %I:%M %p',
                '%d/%m/%Y, %I:%M %p',
                '%m.%d.%y, %H:%M',
                '%d.%m.%y, %H:%M',
            ]
            
            for fmt in timestamp_formats:
                try:
                    parsed_timestamp = pd.to_datetime(timestamp_str.replace('‎', ''), format=fmt)
                    if pd.notna(parsed_timestamp):
                        break
                except ValueError:
                    continue
            
            current_message = {
                "timestamp": parsed_timestamp,
                "sender": sender,
                "message": message
            }
        else:
            if current_message["timestamp"] is not None:
                current_message["message"] += "\n" + line
    
    if current_message["timestamp"] is not None:
        chat_data.append(current_message)
    
    df = pd.DataFrame(chat_data)
    if not df.empty and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    return df

def preprocess_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<media omitted>|you deleted this message\.', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment(text):
    """Analyze sentiment using VADER"""
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# --- Main App ---

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide", initial_sidebar_state="expanded")

st.title("📱 WhatsApp Chat Sentiment Analyzer")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Upload Chat")
    uploaded_file = st.file_uploader("Upload WhatsApp chat (.txt)", type=['txt'])
    
    st.markdown("---")
    st.info("""
    **How to Export WhatsApp Chat:**
    1. Open WhatsApp
    2. Select chat (individual or group)
    3. Click Menu → More → Export chat
    4. Choose "Without Media"
    5. Save as .txt file
    """)

if uploaded_file is not None:
    try:
        file_content = uploaded_file.read().decode('utf-8')
        
        with st.spinner("Parsing chat data..."):
            df = parse_whatsapp_chat(file_content)
        
        if df.empty:
            st.error("Could not parse chat. Please ensure the file format is correct.")
            st.stop()
        
        # Data preprocessing
        with st.spinner("Analyzing sentiment..."):
            df['clean_message'] = df['message'].apply(preprocess_text)
            df['sentiment'] = df['clean_message'].apply(analyze_sentiment)
            df['message_length'] = df['message'].str.len()
            df['word_count'] = df['message'].str.split().str.len()
        
        # Display overview
        st.subheader("📊 Chat Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", len(df))
        with col2:
            st.metric("Unique Senders", df['sender'].nunique())
        with col3:
            st.metric("Date Range", f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        with col4:
            st.metric("Avg Message Length", f"{df['message_length'].mean():.0f} chars")
        
        st.markdown("---")
        
        # Sentiment Analysis
        st.subheader("💭 Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment Distribution
            sentiment_counts = df['sentiment'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
            sentiment_colors = [colors.get(sent, '#3498db') for sent in sentiment_counts.index]
            ax.bar(sentiment_counts.index, sentiment_counts.values, color=sentiment_colors)
            ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Messages')
            ax.set_xlabel('Sentiment')
            for i, v in enumerate(sentiment_counts.values):
                ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Sentiment Percentage
            sentiment_pct = (df['sentiment'].value_counts() / len(df) * 100).round(1)
            fig, ax = plt.subplots(figsize=(8, 5))
            colors_pct = [colors.get(sent, '#3498db') for sent in sentiment_pct.index]
            wedges, texts, autotexts = ax.pie(sentiment_pct.values, labels=sentiment_pct.index, 
                                               autopct='%1.1f%%', colors=colors_pct, startangle=90)
            ax.set_title('Sentiment Percentage', fontsize=14, fontweight='bold')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Activity by Sender
        st.subheader("👥 Activity by Sender")
        
        sender_stats = df.groupby('sender').agg({
            'message': 'count',
            'word_count': 'mean',
            'sentiment': lambda x: (x == 'Positive').sum()
        }).rename(columns={
            'message': 'Messages',
            'word_count': 'Avg Words',
            'sentiment': 'Positive Messages'
        }).sort_values('Messages', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            top_senders = sender_stats.head(10)
            ax.barh(top_senders.index, top_senders['Messages'], color='#3498db')
            ax.set_xlabel('Number of Messages')
            ax.set_title('Top 10 Active Senders', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            for i, v in enumerate(top_senders['Messages'].values):
                ax.text(v + 5, i, str(v), va='center')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.dataframe(sender_stats.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Sentiment Trend Over Time
        st.subheader("📈 Sentiment Trend Over Time")
        
        df_trend = df.copy()
        df_trend['date'] = df_trend['timestamp'].dt.date
        daily_sentiment = df_trend.groupby('date')['sentiment'].apply(
            lambda x: ((x == 'Positive').sum() - (x == 'Negative').sum()) / len(x) * 100
        )
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(daily_sentiment.index, daily_sentiment.values, marker='o', linewidth=2, color='#3498db')
        ax.fill_between(daily_sentiment.index, daily_sentiment.values, alpha=0.3, color='#3498db')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score (%)')
        ax.set_title('Daily Sentiment Trend', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Message Details
        st.subheader("💬 Message Details")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            selected_sender = st.multiselect("Filter by Sender", df['sender'].unique(), default=df['sender'].unique()[:5])
        with col2:
            selected_sentiment = st.multiselect("Filter by Sentiment", ['Positive', 'Negative', 'Neutral'], 
                                               default=['Positive', 'Negative', 'Neutral'])
        
        filtered_df = df[(df['sender'].isin(selected_sender)) & (df['sentiment'].isin(selected_sentiment))].copy()
        filtered_df = filtered_df.sort_values('timestamp', ascending=False)
        
        st.dataframe(filtered_df[['timestamp', 'sender', 'message', 'sentiment', 'word_count']].head(50), 
                    use_container_width=True)
        
        st.markdown("---")
        
        # Word Statistics
        st.subheader("📝 Word Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Words", df['word_count'].sum())
        with col2:
            st.metric("Avg Words/Message", f"{df['word_count'].mean():.1f}")
        with col3:
            st.metric("Max Words/Message", df['word_count'].max())
        
        # Word length distribution
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(df['word_count'], bins=30, color='#3498db', edgecolor='black')
        ax.set_xlabel('Words per Message')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Message Length (Words)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure the file is a valid WhatsApp chat export in .txt format.")
else:
    st.info("👈 Upload a WhatsApp chat file to get started!")
    
    # Show example
    st.subheader("Example Output")
    st.markdown("""
    This app will analyze your WhatsApp chat and provide:
    - **Sentiment Analysis**: Positive, Negative, or Neutral message classification
    - **Sender Statistics**: Message counts, activity levels, and positive message ratios
    - **Trends**: Sentiment changes over time
    - **Detailed View**: Individual messages with sentiment labels
    - **Word Statistics**: Message length and complexity analysis
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
</div>
""", unsafe_allow_html=True)
