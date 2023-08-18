from __future__ import unicode_literals
from pickle import FALSE
from flask import Flask, render_template, request
import youtube_dl
import whisper
import warnings

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

def safe_to_text_file(text_, nameOfSong):
    result_file = open(nameOfSong + ' text_of_video.text', 'w')
    result_file.write(text_['text'])
    result_file.close()
    return 'text_of_video.text'

@app.route('/')
def index():
    print("hello")
    return(render_template('index.html'))

@app.route('/answer', methods=['POST'])
def answer():
    url_ = request.form.get('searchbox')
    filename_, songname_ = save_to_mp3(url_)
    print(filename_)
    print(songname_)
    text = convert_file_to_text(filename_)
    nameOfTextFile = safe_to_text_file(text, songname_)


    return(render_template('answer.html', songname=songname_))

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
