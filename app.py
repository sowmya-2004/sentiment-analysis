from flask import Flask, render_template, request, jsonify
from flask import request
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__, template_folder='templ')

def load_reviews(file):
    reviews = file.read().decode('utf-8').splitlines()
    return reviews

def analyze_sentiments(reviews):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [sia.polarity_scores(review)["compound"] for review in reviews]
    return sentiment_scores

def count_sentiments(sentiment_scores):
    sentiment_counts = {'positive': sum(score > 0 for score in sentiment_scores),
                        'neutral': sum(score == 0 for score in sentiment_scores),
                        'negative': sum(score < 0 for score in sentiment_scores)}
    return sentiment_counts

def plot_to_base64(figure):
    buf = BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()  # Clear the figure
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

# ... (previous code)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            reviews = load_reviews(file)
            if reviews:
                sentiment_scores = analyze_sentiments(reviews)
                sentiment_counts = count_sentiments(sentiment_scores)
                plt.figure()
                labels = list(sentiment_counts.keys())
                sizes = list(sentiment_counts.values())
                colors = ['gold', 'lightcoral', 'lightskyblue']
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
                plt.axis('equal')
                plt.title('Sentiment Distribution (Predicted)')
                img_base64 = plot_to_base64(plt)
                return jsonify({'image': img_base64, 'sentiment_counts': sentiment_counts})
            else:
                return jsonify({'error': 'Empty file'})
        else:
            return jsonify({'error': 'No selected file'})
    elif 'text' in request.form:
        text = request.form['text']
        sentiment_scores = analyze_sentiments([text])
        sentiment_counts = count_sentiments(sentiment_scores)
        plt.figure()
        labels = list(sentiment_counts.keys())
        sizes = list(sentiment_counts.values())
        colors = ['gold', 'lightcoral', 'lightskyblue']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title('Sentiment Distribution (Predicted)')
        img_base64 = plot_to_base64(plt)
        return jsonify({'image': img_base64, 'sentiment_counts': sentiment_counts})
    else:
        return jsonify({'error': 'No file or text provided'})


if __name__ == '__main__':
    app.run(debug=True)
