import os
from OCR import get_text_from_img
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.template_folder = os.path.join(app.instance_path, 'template')

UPLOAD_FOLDER = os.path.join(app.instance_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home_page():
    return "Hello World!"

@app.route('/Upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        if 'file' not in request.files:
            return render_template('upload.html', msg='No file selected')

        file = request.files['file']

        if file.filename == '':
            return render_template('upload.html', msg='No file selected')

        if file and allowed_file(file.filename):

            filename = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(filename)

            extracted_text = get_text_from_img(filename)

            return render_template('upload.html',
                                   msg='Successfully processed',
                                   extracted_text=extracted_text,
                                   img_src=filename)

    elif request.method == 'GET':
        return render_template('upload.html')


if __name__ == '__main__':
    app.run()
