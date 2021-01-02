from DataPreprocessing import dataProcessing
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

from NLPFunctions import remove_noise

def createClassifier():
    processedData = dataProcessing()
    classifier = NaiveBayesClassifier.train(processedData)
    return classifier


'''
    random model testing!!!!!!!!!!!!
   

    choice = 'y'

    while (choice == 'y'):
        custom_tweet = input("Enter your comment : ") 
        custom_tokens = remove_noise(word_tokenize(custom_tweet))
        print(classifier.classify(dict([token, True] for token in custom_tokens)))
        choice = input("press y to continue n to exit ")

createClassifier()
'''