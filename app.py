from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from nltk.sentiment import SentimentIntensityAnalyzer
from googletrans import Translator

app = Flask(__name__, template_folder='templates')

def load_reviews(file):
    reviews = file.read().decode('utf-8').splitlines()
    return reviews

def translate_to_english(text):
    translator = Translator()
    translated = translator.translate(text, src='auto', dest='en')
    return translated.text

def analyze_sentiments(reviews):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [sia.polarity_scores(review)["compound"] for review in reviews]
    return sentiment_scores

def count_sentiments(sentiment_scores):
    sentiment_counts = {'positive': sum(score > 0 for score in sentiment_scores),
                        'neutral': sum(score == 0 for score in sentiment_scores),
                        'negative': sum(score < 0 for score in sentiment_scores)}
    return sentiment_counts

@app.route('/')
def index():
    return render_template('new/index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    reviews = load_reviews(file)
    if not reviews:
        return jsonify({'error': 'Empty file'})

    print("Contents of reviews file:")
    print(reviews)

    # Translate reviews to English
    reviews_english = [translate_to_english(review) for review in reviews]

    # Analyze sentiments using polarity scores
    sentiment_scores = analyze_sentiments(reviews_english)

    print("Sentiment Scores:")
    print(sentiment_scores)

    # Count the number of reviews for each sentiment
    sentiment_counts = count_sentiments(sentiment_scores)

    print("Sentiment Counts:")
    print(sentiment_counts)

    # Create a new figure and pie chart based on sentiment scores
    plt.figure()
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())

    colors = ['gold', 'lightcoral', 'lightskyblue']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Sentiment Distribution (Predicted)')

    img_base64 = plot_to_base64(plt)

    return jsonify({'image': img_base64, 'sentiment_counts': sentiment_counts})

def plot_to_base64(figure):
    buf = BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()  # Clear the figure
    return img_base64

if __name__ == '__main__':
    app.run(debug=True)
