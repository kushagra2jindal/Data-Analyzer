from classifier import createClassifier
from flask import Flask
from flask import request, jsonify

from nltk.tokenize import word_tokenize
from NLPFunctions import remove_noise

app = Flask(__name__)

@app.route('/api/v1/classifycomment', methods=['GET'])
def classify_comment():
    if 'query_string' in request.args:
        query = request.args['query_string']
        custom_tokens = remove_noise(word_tokenize(query))
        return classifier.classify(dict([token, True] for token in custom_tokens))
    else:
        return "Error: No query_string field provided. Please specify an query."



if __name__ == '__main__':
    classifier = createClassifier()
    app.run()