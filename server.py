from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
    print("hello")
    return(render_template('index.html'))

@app.route('/answer', methods=['POST'])
def answer():
    url_ = request.form.get("searchbox")
    answer_ = url_
    return(render_template('answer.html', answer=answer_))

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
