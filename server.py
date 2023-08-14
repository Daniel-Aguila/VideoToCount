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
    return downloader.prepare_filename(downloader.extract_info(url, download=False)).replace(".m4a", ".mp3")

def convert_file_to_text(file):
    model = whisper.load_model("base")
    result = model.transcribe(file, fp16=FALSE)
    return result


@app.route('/')
def index():
    print("hello")
    return(render_template('index.html'))

@app.route('/answer', methods=['POST'])
def answer():
    url_ = request.form.get('searchbox')
    filename = save_to_mp3(url_)
    print(filename)
    result = convert_file_to_text(filename)
    result_file = open('text_of_video.text', 'w')
    result_file.write(result['text'])
    result_file.close()

    answer_ = url_

    return(render_template('answer.html', answer=answer_))

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
