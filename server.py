from __future__ import unicode_literals
from statistics import multimode
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from sklearn.model_selection import train_test_split

pd.set_option('display.max_colwidth',0)
from pickle import FALSE
from xml.dom import WRONG_DOCUMENT_ERR
from flask import Flask, render_template, request
import youtube_dl
import whisper
import warnings
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re


warnings.filterwarnings("ignore")

app = Flask(__name__)

def save_to_mp3(url):
    options={
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    with youtube_dl.YoutubeDL(options) as downloader:
        downloader.download(["" + url + ""])
    name_ = downloader.extract_info(url, download=False)
    return downloader.prepare_filename(downloader.extract_info(url, download=False)).replace(".webm",".mp3"), name_['title']

def convert_file_to_text(file):
    model = whisper.load_model("base")
    result = model.transcribe(file, fp16=FALSE)
    return result

#def preprocessAndWrite(text_,nameOfSong):
    #full_name_of_text = nameOfSong + ' text_of_video.text'
    #unprocessed_text = text_['text']

    #Preprocess

    #sentences = sent_tokenize(unprocessed_text)
    #lemmatizer = WordNetLemmatizer()

    #for i in range(len(sentences)):
    #    processed_text = re.sub('[^a-zA-Z]',' ', sentences[i]) #replaces all the characters except for a-z A-Z characters into spaces
    #    processed_text = processed_text.lower()
    #    processed_text = processed_text.split('.')
    #    tokens_without_sw = [lemmatizer.lemmatize(word) for word in processed_text if word not in set(stopwords.words('english'))]
    #
    #    sentences[i] = (" ").join(tokens_without_sw)

    #result_file = open(full_name_of_text, 'w')
    #result_file.writelines(unprocessed_text)
    #print(unprocessed_text)
    #result_file.close()
    #return full_name_of_text
def topGenre(songLyrics):
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    sequence_to_classify = songLyrics['text']
    candidate_labels=[
        "Joy", "Sadness", "Anger","Fear","Surprise","Disgust","Love","Excitement","Confidence","Curiosity","Shame","Hope","Loneliness","Satisfaction"
        ]
    output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
    return output['labels'][0]

@app.route('/')
def index():
    print("hello")
    return(render_template('index.html'))

@app.route('/answer', methods=['POST'])
def answer():
    url_ = request.form.get('searchbox')
    filename_, songname_ = save_to_mp3(url_)
    text = convert_file_to_text(filename_)
    songgenre_ = topGenre(text)

    embed_url = url_.replace("watch?=","embed/")
    return(render_template('answer.html', songname=songname_, songgenre=songgenre_, youtubelink=embed_url))

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
