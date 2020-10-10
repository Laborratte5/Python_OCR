import os
import Upload
import Defines
import ProcessImage
from flask import Flask, render_template, request

app = Flask(__name__)
app.template_folder = os.path.join(app.instance_path, 'template')

ProcessImage.InitProcessImage(app.instance_path)
Upload.InitUpload(app.instance_path)

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        if 'file' not in request.files:
            return render_template('upload.html', lngs=Defines.lngs, msg='No file selected')

        filename = Upload.Upload( request.files['file'] )

        if filename == '':
            return render_template('upload.html', lngs=Defines.lngs, msg='No file selected')

        extracted_text, summary, threshold = ProcessImage.ProcessImage(filename, request.form['lng'])

        # TODO Correct mapping on slider by interpolating between min and max value based on slider value
        # TODO Update Requirements
        # TODO Tesseract does not work with Deu selected, idk why

        render_text = '\n'
        for sentence, value in summary:
            render_text += '[`' + sentence + '`,' + str(value) + '],\n'

        return render_template('upload.html', 
                                lngs=Defines.lngs,
                                msg='Successfully processed',
                                threshold=threshold,
                                extracted_text=render_text,
                                img_src=filename)

    elif request.method == 'GET':
        return render_template('upload.html', lngs=Defines.lngs)


if __name__ == '__main__':
    app.run()
