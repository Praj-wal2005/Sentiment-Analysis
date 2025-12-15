import base64
import io
import logging
import re
import pandas as pd
import matplotlib
# set backend to Agg before importing pyplot to avoid GUI errors on servers
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from flask import Flask, request, jsonify, render_template, send_file
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch

# Configure logging to see progress in terminal
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# --- 1. LOAD ALL AI MODELS (The "NLP Techniques") ---
logging.info("⏳ Loading AI Models... This may take 2-3 minutes on the first run.")

# Model A: Sentiment Analysis (The Core Requirement)
# We use a model trained on social media (Tweets) for accurate 3-way classification.
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

# Model B: Emotion Detection (Advanced Feature)
# Detects: joy, sadness, anger, fear, surprise, neutral, disgust
emotion_classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    return_all_scores=True
)

# Model C: Named Entity Recognition (NER) (Advanced Feature)
# Identifies 'Entities' like Organizations (Apple), Locations (India), Persons (Elon Musk)
ner_classifier = pipeline(
    "ner", 
    model="dslim/bert-base-NER", 
    aggregation_strategy="simple"
)

logging.info("✅ All Models Loaded Successfully!")

# --- HELPER FUNCTIONS ---

def clean_text(text):
    """Basic NLP preprocessing to clean noise from text."""
    if not isinstance(text, str): return ""
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'http\S+', '', text) # Remove URLs
    return text.strip()

def generate_wordcloud_base64(text):
    """Generates a visual WordCloud image from text."""
    if not text or len(text.split()) < 3:
        return None
    
    # Generate the cloud image
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Save image to memory buffer (RAM) instead of a file
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    
    # Convert to base64 string for HTML display
    return base64.b64encode(img.getvalue()).decode('utf-8')

def analyze_single_text(text):
    """Runs all 3 AI models on one piece of text."""
    cleaned = clean_text(text)
    
    if not cleaned:
        return None

    # 1. Run Sentiment Analysis
    inputs = sentiment_tokenizer(cleaned, return_tensors="pt")
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
    scores = F.softmax(logits, dim=1).numpy()[0]
    
    # Map scores to labels
    sentiment_dict = {
        'negative': float(scores[0]),
        'neutral':  float(scores[1]),
        'positive': float(scores[2])
    }
    winner_sentiment = max(sentiment_dict, key=sentiment_dict.get).upper()

    # 2. Run Emotion Analysis
    emotions_raw = emotion_classifier(cleaned)[0]
    # Sort emotions by highest score
    emotions_sorted = sorted(emotions_raw, key=lambda x: x['score'], reverse=True)
    top_emotion = emotions_sorted[0]['label']

    # 3. Run NER Analysis (Extract entities with >50% confidence)
    entities_raw = ner_classifier(cleaned)
    entities = [
        {'word': e['word'], 'entity': e['entity_group'], 'score': float(e['score'])}
        for e in entities_raw if e['score'] > 0.50 
    ]

    return {
        'sentiment': winner_sentiment,
        'sentiment_scores': sentiment_dict,
        'top_emotion': top_emotion,
        'entities': entities
    }

# --- SERVER ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Handle Single Text Analysis
    data = request.json
    text = data.get('text', '')
    
    result = analyze_single_text(text)
    wc_image = generate_wordcloud_base64(text)
    
    if result:
        return jsonify({
            'status': 'success',
            'data': result,
            'wordcloud': wc_image
        })
    else:
        return jsonify({'status': 'error', 'message': 'Invalid text'})

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    # Handle Bulk CSV Upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'})

    try:
        df = pd.read_csv(file)
        # Verify columns
        text_col = 'text' if 'text' in df.columns else df.columns[0]
        
        # Analyze first 50 rows for demonstration
        results = []
        for text in df[text_col].astype(str).head(50):
            res = analyze_single_text(text)
            if res:
                results.append({
                    'Original Text': text,
                    'Sentiment': res['sentiment'],
                    'Top Emotion': res['top_emotion'],
                    'Confidence': res['sentiment_scores'][res['sentiment'].lower()]
                })
        
        # Save results to a new CSV in memory
        result_df = pd.DataFrame(results)
        output = io.BytesIO()
        result_df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output, 
            mimetype="text/csv", 
            as_attachment=True, 
            download_name="analyzed_reviews.csv"
        )

    except Exception as e:
        logging.error(f"CSV Error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)