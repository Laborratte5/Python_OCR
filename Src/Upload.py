
import os
import Defines
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = ''

def InitUpload(instance_path):
    global UPLOAD_FOLDER
    UPLOAD_FOLDER = os.path.join(instance_path, Defines.uploads_path)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Defines.allowed_extensions

def Upload(file):

    if file.filename == '':
        return ''

    if file and allowed_file(file.filename):

        filename = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(Defines.temp_file_path)
        
        return filename

    return ''