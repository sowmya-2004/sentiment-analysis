from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from nltk.sentiment import SentimentIntensityAnalyzer
from googletrans import Translator

app = Flask(__name__, template_folder='templ')

def load_reviews(file):
    reviews = file.read().decode('utf-8').splitlines()
    return reviews


def translate_to_english(reviews):
    translator = Translator()
    english_reviews = []

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
    return render_template('index.html')

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
    english_reviews = translate_to_english(reviews)
    print("Translated Reviews:")
    print(english_reviews)

    # Analyze sentiments using polarity scores
    sentiment_scores = analyze_sentiments(english_reviews)

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
