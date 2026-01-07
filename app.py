from flask import Flask, render_template, request
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

ps = PorterStemmer()

def preprocess_text(text):
    content = re.sub('[^a-zA-Z]', ' ', text)
    content = content.lower().split()
    content = [ps.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None

    if request.method == 'POST':
        title = request.form['news_title']
        text = request.form['news_text']
        combined = title + ' ' + text

        processed = preprocess_text(combined)
        vectorized = vectorizer.transform([processed])
        pred = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]

        prediction = 'FAKE' if pred == 1 else 'REAL'
        confidence = round(max(prob) * 100, 2)

    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
