from classifier import createClassifier
from flask import Flask
from flask import request, jsonify

from nltk.tokenize import word_tokenize
from NLPFunctions import remove_noise

app = Flask(__name__)

@app.route('/api/v1/emotionclassifier', methods=['GET'])
def classify_comment():
    if 'query_string' in request.args:
        query = request.args['query_string']
        custom_tokens = remove_noise(word_tokenize(query))
        result = classifier.classify(dict([token, True] for token in custom_tokens))
        return jsonify([{'result':result}])
    else:
        return jsonify([{"Error" : "No query_string field provided. Please specify an query."}])



if __name__ == '__main__':
    classifier = createClassifier()
    app.run()