from flask import Flask, request, render_template, redirect, url_for
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import os
import numpy as np
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def check_tampering(original_path, tampered_path, size=(250, 160)):
    original = cv2.imread(original_path)
    tampered = cv2.imread(tampered_path)

    # Resize images
    original = cv2.resize(original, size)
    tampered = cv2.resize(tampered, size)

    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return score, original, tampered, diff, thresh


@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files or 'tampered_file' not in request.files:
            return redirect(request.url)
        original_file = request.files['file']
        tampered_file = request.files['tampered_file']
        if original_file.filename == '' or tampered_file.filename == '':
            return redirect(request.url)
        if original_file and tampered_file:
            original_filename = secure_filename(original_file.filename)
            tampered_filename = secure_filename(tampered_file.filename)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            tampered_path = os.path.join(app.config['UPLOAD_FOLDER'], tampered_filename)
            original_file.save(original_path)
            tampered_file.save(tampered_path)
            score, original, tampered, diff, thresh = check_tampering(original_path, tampered_path)

            # Save the resulting images to display
            result_original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_original.png')
            result_tampered_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_tampered.png')
            result_diff_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_diff.png')
            result_thresh_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_thresh.png')

            cv2.imwrite(result_original_path, original)
            cv2.imwrite(result_tampered_path, tampered)
            cv2.imwrite(result_diff_path, diff)
            cv2.imwrite(result_thresh_path, thresh)

            return render_template('result.html', score=score, result_original_path=result_original_path,
                                   result_tampered_path=result_tampered_path, result_diff_path=result_diff_path,
                                   result_thresh_path=result_thresh_path)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
