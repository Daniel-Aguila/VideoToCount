from __future__ import unicode_literals
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import watson_nlp
import watson_nlp.data_model as dm

from sklearn.model_selection import train_test_split
from watson_core.toolkit import fileio
from watson_core.toolkit.quality_evaluation import QualityEvaluator, EvalTypes

from watson_nlp.blocks.classification.bert import BERT
from watson_nlp.blocks.classification.cnn import CNN
from watson_core.data_model.streams.resolver import DataStreamResolver
from watson_core.data_model.streams.resolver import DataStream
from watson_nlp.blocks.classification.svm import SVM
from watson_nlp.blocks.vectorization.tfidf import TFIDF

pd.set_option('display.max_colwidth',0)

block_models = watson_nlp.get_models().get_alias_models()
workflow_models = watson_nlp.get_workflows().get_alias_models()

from concurrent.futures import process
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

def preprocessAndWrite(text_,nameOfSong):
    full_name_of_text = nameOfSong + ' text_of_video.text'
    unprocessed_text = text_['text']

    #Preprocess
    processed_text = unprocessed_text.lower()
    processed_text = processed_text.replace(r",","")
    processed_text = processed_text.replace(r"!","")
    processed_text = processed_text.replace(r".","")
    processed_text = processed_text.replace(r"1","")
    processed_text = processed_text.replace(r"2","")
    processed_text = processed_text.replace(r"3","")
    processed_text = processed_text.replace(r"4","")
    processed_text = processed_text.replace(r"5","")
    processed_text = processed_text.replace(r"6","")
    processed_text = processed_text.replace(r"7","")
    processed_text = processed_text.replace(r"8","")
    processed_text = processed_text.replace(r"9","")
    processed_text = processed_text.replace(r"?","")
    processed_text = processed_text.replace(r"(can't|cannot)","can not")
    processed_text = processed_text.replace(r"i'm","i am")
    processed_text = processed_text.replace(r"what's","what is")
    processed_text = processed_text.replace(r"i've","i have")
    processed_text = processed_text.replace(r"she's","she is")
    processed_text = processed_text.replace(r"he's","he is")
    processed_text = processed_text.replace(r"it's","it is")
    processed_text = processed_text.replace(r"there's","there is")
    processed_text = processed_text.replace(r"n't"," not")
    processed_text = processed_text.replace(r"we're","we are")
    processed_text = processed_text.replace(r"is's","is")
    processed_text = processed_text.replace(r"-"," ")
    processed_text = processed_text.replace(r"they'll","they will")
    processed_text = processed_text.replace(r"they're","they are")
    processed_text = processed_text.replace(r"let's","let us")
    processed_text = processed_text.replace(r"that's","that is")
    processed_text = processed_text.replace(r"  "," ")
    processed_text = processed_text.replace(r"'","")

    text_tokens = word_tokenize(processed_text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    

    filtered_sentence = (" ").join(tokens_without_sw)

    result_file = open(full_name_of_text, 'w')
    result_file.write(filtered_sentence)
    print(filtered_sentence)
    result_file.close()
    return full_name_of_text
    

@app.route('/')
def index():
    print("hello")
    return(render_template('index.html'))

@app.route('/answer', methods=['POST'])
def answer():
    url_ = request.form.get('searchbox')
    filename_, songname_ = save_to_mp3(url_)
    text = convert_file_to_text(filename_)
    preprocessAndWrite(text,songname_)


    return(render_template('answer.html', songname=songname_))

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
