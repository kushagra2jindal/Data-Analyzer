from nltk.tag import pos_tag
import pandas as pd 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
from NLPFunctions import remove_noise, get_sentence_for_model

dataset_Happy = pd.read_csv('Dataset/EmotionalHappy.csv')['Sentence']
dataset_Anger = pd.read_csv('Dataset/EmotionalAnger.csv')['Sentence']
dataset_Disgust = pd.read_csv('Dataset/EmotionalDisgust.csv')['Sentence']
dataset_Sad = pd.read_csv('Dataset/EmotionalSad.csv')['Sentence']
dataset_Shame = pd.read_csv('Dataset/EmotionalShame.csv')['Sentence']
dataset_Surprise = pd.read_csv('Dataset/EmotionalSurprise.csv')['Sentence']

stop_words = stopwords.words('english')


def dataProcessing():

    #sentences = dataset['Sentence']

    token_list_happy = []
    token_list_anger = []
    token_list_disgust = []
    token_list_sad = []
    token_list_shame = []
    token_list_surprise = []

    for sentence in dataset_Happy:
        token_list_happy.append(word_tokenize(sentence))
    for sentence in dataset_Anger:
        token_list_anger.append(word_tokenize(sentence))
    for sentence in dataset_Disgust:
        token_list_disgust.append(word_tokenize(sentence))
    for sentence in dataset_Sad:
        token_list_sad.append(word_tokenize(sentence))
    for sentence in dataset_Shame:
        token_list_shame.append(word_tokenize(sentence))
    for sentence in dataset_Surprise:
        token_list_surprise.append(word_tokenize(sentence))

    clean_token_list_happy = []
    clean_token_list_anger = []
    clean_token_list_disgust = []
    clean_token_list_sad = []
    clean_token_list_shame = []
    clean_token_list_surprise = []

    for tokens in token_list_happy:
        clean_token_list_happy.append(remove_noise(tokens, stop_words))
    for tokens in token_list_anger:
        clean_token_list_anger.append(remove_noise(tokens, stop_words))
    for tokens in token_list_disgust:
        clean_token_list_disgust.append(remove_noise(tokens, stop_words))
    for tokens in token_list_sad:
        clean_token_list_sad.append(remove_noise(tokens, stop_words))
    for tokens in token_list_shame:
        clean_token_list_shame.append(remove_noise(tokens, stop_words))
    for tokens in token_list_surprise:
        clean_token_list_surprise.append(remove_noise(tokens, stop_words))


    Tokens_for_model_happy = get_sentence_for_model(clean_token_list_happy)
    Tokens_for_model_anger = get_sentence_for_model(clean_token_list_anger)
    Tokens_for_model_disgust = get_sentence_for_model(clean_token_list_disgust)
    Tokens_for_model_sad = get_sentence_for_model(clean_token_list_sad)
    Tokens_for_model_shame = get_sentence_for_model(clean_token_list_shame)
    Tokens_for_model_surprise = get_sentence_for_model(clean_token_list_surprise)


    Happy_dataset = [(tweet_dict, "Happy")
                    for tweet_dict in Tokens_for_model_happy]
    Anger_dataset = [(tweet_dict, "Anger")
                    for tweet_dict in Tokens_for_model_anger]
    Disgust_dataset = [(tweet_dict, "Disgust")
                    for tweet_dict in Tokens_for_model_disgust]
    Sad_dataset = [(tweet_dict, "Sad")
                    for tweet_dict in Tokens_for_model_sad]
    Shame_dataset = [(tweet_dict, "Shame")
                    for tweet_dict in Tokens_for_model_shame]
    Surprise_dataset = [(tweet_dict, "Surprise")
                    for tweet_dict in Tokens_for_model_surprise]   

    dataset = Happy_dataset + Anger_dataset + Disgust_dataset + Sad_dataset + Shame_dataset + Surprise_dataset
    random.shuffle(dataset)

    train_data = dataset[:1300]
    return (train_data)



dataProcessing()