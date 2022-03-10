import os

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

from get_prediction import make_prediction

app = Flask(__name__, template_folder='Template')
Bootstrap(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join('static', uploaded_file.filename)
            uploaded_file.save(image_path)
            class_name = make_prediction(image_path)
            result = {
                'class_name': class_name,
                'image_path': image_path,
            }
            return render_template('result.html', result=result)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000", debug=True)
