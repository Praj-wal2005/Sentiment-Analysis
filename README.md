# AI Customer Insights Hub

Advanced NLP Dashboard for Sentiment, Emotion, and Entity Analysis

The *AI Customer Insights Hub* is a full-stack Flask application designed to analyze customer feedback using state-of-the-art Deep Learning models. It goes beyond simple sentiment analysis by detecting specific emotions (joy, anger, etc.) and extracting named entities (people, brands) from text.

#  Key Features

Multi-Model Analysis:
    * Sentiment Analysis:Uses `twitter-roberta-base-sentiment` to classify text as Positive, Neutral, or Negative.
    * Emotion Detection: Implements `distilroberta-base` to detect nuances like Joy, Sadness, Anger, and Fear.
    * Named Entity Recognition (NER): Uses BERT to automatically identify Organizations, Locations, and Persons within the text.
Voice-to-Text Support: Integrated Web Speech API allows users to analyze sentiment via microphone input.
Bulk CSV Processing: Users can upload a CSV file of reviews to process thousands of rows instantly and download the analyzed results.
Visualizations:Generates dynamic Word Clouds and real-time confidence score badges.

# Tech Stack

Backend:Python, Flask
AI/ML:Hugging Face Transformers, PyTorch
Data Processing:Pandas, NumPy
Visualization: Matplotlib, WordCloud
Frontend:** HTML5, CSS3, Vanilla JavaScript

# Installation & Usage

1.  Clone the repository:bash
    git clone [https://github.com/yourusername/ai-customer-insights.git](https://github.com/yourusername/ai-customer-insights.git)
    cd ai-customer-insights

2. Install dependencies:bash
    pip install -r requirements.txt
    

3. Run the application:bash
    python app.py

4.  Access the Dashboard:
    Open your browser and navigate to `http://localhost:5000`.
