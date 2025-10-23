from flask import Flask, render_template, request
from textblob import TextBlob

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    polarity = None
    subjectivity = None

    if request.method == 'POST':
        text = request.form['text']
        if text.strip() != "":
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Classify sentiment
            if polarity > 0:
                result = "Positive 😊"
            elif polarity < 0:
                result = "Negative 😞"
            else:
                result = "Neutral 😐"
        else:
            result = "Please enter some text!"

    return render_template('index.html', result=result, polarity=polarity, subjectivity=subjectivity)

if __name__ == '__main__':
    app.run(debug=True)
