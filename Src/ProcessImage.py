
import os
import csv
import Defines
from OCR import get_text_from_img
from Summarization import run_summarization

CSV_File = ''

def InitProcessImage(instance_path):
    global CSV_File
    CSV_File = os.path.join(instance_path, Defines.csv_path)

    # make sure, that csv file has been created
    with open(CSV_File, "a", encoding="utf-8") as csv_file:
        pass

# removes empty lines from csv file
# TODO find better Fix
def CleanCSVFile():
    with open(CSV_File, 'r+') as fd:
        lines = fd.readlines()
        fd.seek(0)
        fd.writelines(line for line in lines if line.strip())
        fd.truncate()

def getFileNameForCSV(basefilename):
    def FilenameInCSV(basefilename):
        with open(CSV_File, "r+", encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=Defines.csv_delimiter)
            for row in csv_reader:
                if len(row) > 0 and row[0] == basefilename:
                    return True
        return False

    idx = 0
    csvfilename = basefilename
    while FilenameInCSV(csvfilename):
        idx += 1
        csvfilename = basefilename + str(idx)

    return csvfilename

def ProcessImage(filename, lng):
    basefilename, extension = os.path.splitext(filename)

    csvfilename = getFileNameForCSV(basefilename)
    
    os.rename(Defines.temp_file_path, csvfilename + extension)

    extracted_text = get_text_from_img(csvfilename + extension, lng)

    summary, keywords, threshold = run_summarization(extracted_text, Defines.lng_to_NLTKlng[lng])

    with open(CSV_File, "a", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=Defines.csv_delimiter)
        csv_writer.writerow([ csvfilename + extension, *keywords ])

    CleanCSVFile()

    with open(csvfilename + ".txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    with open(csvfilename + ".summ", "w", encoding="utf-8") as f:
        for sentence, _ in summary:
            f.write(sentence + "\n")
    
    return extracted_text, summary, threshold
