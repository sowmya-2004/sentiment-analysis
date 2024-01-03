from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from nltk.sentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import threading

app = Flask(__name__, template_folder='templ')
lock = threading.Lock()

def load_reviews(file):
    reviews = file.read().decode('utf-8').splitlines()
    return reviews

def translate_to_english(reviews):
    translator = Translator()
    english_reviews = []

    # Ensure reviews are treated as a list
    if not isinstance(reviews, list):
        reviews = [reviews]

    for review in reviews:
        try:
            translation = translator.translate(review)
            if translation.text:
                english_reviews.append(translation.text)
            else:
                # If translation is empty, use the original review
                english_reviews.append(review)
        except Exception as e:
            print(f"Translation error: {e}")
            # If an error occurs during translation, use the original review
            english_reviews.append(review)

    return english_reviews

def analyze_sentiments(review):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(review)["compound"]
    return sentiment_score

def classify_review(sentiment_score):
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

def generate_plot(sentiment_counts):
    plt.figure()
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())

    colors = ['gold', 'lightcoral', 'lightskyblue']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Sentiment Distribution (Predicted)')

    img_base64 = plot_to_base64(plt.gcf())  # gcf() retrieves the current figure

    return img_base64

def analyze_and_plot(file):
    reviews = load_reviews(file)
    if not reviews or all(not review.strip() for review in reviews):
        return jsonify({'error': 'No valid reviews in the file'})

    print("Contents of reviews file:")
    print(reviews)

    sentiment_counts = {
        'positive': 0,
        'neutral': 0,
        'negative': 0
    }

    for review in reviews:
        stripped_review = review.strip()
        if stripped_review:
            sentiment_score = analyze_sentiments(stripped_review)
            classification = classify_review(sentiment_score)
            sentiment_counts[classification] += 1

    print("Sentiment Counts:")
    print(sentiment_counts)

    img_base64 = generate_plot(sentiment_counts)

    return jsonify({'image': img_base64, 'sentiment_counts': sentiment_counts})

def plot_to_base64(figure):
    buf = BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(figure)  # Clear the figure
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    return analyze_and_plot(file)

if __name__ == '__main__':
    app.run(debug=True)
