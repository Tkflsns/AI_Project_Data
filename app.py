from flask import Flask, request, jsonify
import os
import shutil
import base64

app = Flask(__name__)

UPLOAD_FOLDER = './input'
OUTPUT_FOLDER = './output'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 파일 업로드 및 응답으로 이미지와 텍스트 전송
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        input_filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_filepath)

        output_filepath = os.path.join(OUTPUT_FOLDER, filename)
        shutil.copy(input_filepath, output_filepath)

        # 텍스트 파일 생성
        text_filename = f"{os.path.splitext(filename)[0]}.txt"
        text_filepath = os.path.join(OUTPUT_FOLDER, text_filename)
        with open(text_filepath, 'w') as f:
            f.write('12가3456')

        # 이미지 파일을 Base64로 인코딩
        with open(output_filepath, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # 텍스트 파일 내용을 읽기
        text_content = ""
        if os.path.exists(text_filepath):
            with open(text_filepath, 'r') as text_file:
                text_content = text_file.read()

        # JSON 응답으로 이미지(Base64)와 텍스트 내용 전송
        response = {
            'message': 'File uploaded and processed successfully',
            'image': encoded_image,  # Base64 인코딩된 이미지 데이터
            'text': text_content     # 텍스트 파일의 내용
        }

        return jsonify(response), 200

    return jsonify({'error': 'Invalid file format'}), 400


if __name__ == '__main__':
    app.run(debug=True)
