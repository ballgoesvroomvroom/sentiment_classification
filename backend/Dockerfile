FROM python:3.10

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# enter otherwise file refs would fail at runtime
WORKDIR /app/src

# install nltk data at build time
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng')"

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]