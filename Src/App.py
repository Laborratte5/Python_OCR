import os
import csv
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


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_term = request.form['search']
        
        results = []
        with open(ProcessImage.CSV_File, "r+", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=Defines.csv_delimiter)
            for row in csv_reader:
                for token in row:
                    if search_term in token:
                        file = row[0]
                        summery_path = os.path.splitext(file)[0] + ".summ"
                        with open(summery_path, "r+", encoding="utf-8") as f:
                            summery = "".join( f.readlines() )
                        results.append((file, summery))
                        break

        return render_template('search.html', results=results)
    else:
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

        render_text = '\n'
        for sentence, value in summary:
            # Format to be displayed in js on client
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
