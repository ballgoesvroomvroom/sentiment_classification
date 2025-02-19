"""
flask server
"""
from flask import Flask, request, jsonify
from model import FinalEstimator

app = Flask(__name__)
Predictor = FinalEstimator()

@app.route('/api/sentiment', methods=['POST'])
def sentiment_analysis():
	data = request.get_json()
	if not data or 'text' not in data:
		return jsonify({"error": "Missing 'text' field in request body"}), 400

	sentiment, tokens, neg_proba, pos_proba = Predictor.predict(data["text"])

	response = {
		"sentiment": sentiment,
		"tokens": tokens,
		"neg": neg_proba,
		"pos": pos_proba
	}
	return jsonify(response)