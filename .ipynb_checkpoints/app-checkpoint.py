from flask import Flask, request, jsonify
from image_processing_pipeline import process_image  # process_images 함수 사용
import os
import shutil
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'input/'
OUTPUT_FOLDER = 'output/'
PROCESSED_FOLDER = 'Boxed_crop/gray'  # 처리된 이미지가 저장되는 폴더 경로

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 파일 업로드 및 처리 후 이미지와 텍스트 전송
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

        # 이미지 처리 후 결과 텍스트 파일 경로 반환
        image_file, text_filepath = process_image(input_filepath)
        print(image_file)
        # 처리된 이미지 파일 경로 설정 (Boxed_crop/gray 디렉토리에서 이름이 같은 파일)
        processed_image_filename = os.path.basename(image_file)
        print(processed_image_filename)
        processed_image_path = os.path.join(OUTPUT_FOLDER, PROCESSED_FOLDER, processed_image_filename)
        print(processed_image_path)
        print(text_filepath)

        # 이미지 파일을 Base64로 인코딩
        if os.path.exists(processed_image_path):
            with open(processed_image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        else:
            return jsonify({'error': 'Processed image not found'}), 404

        # 텍스트 파일 내용을 읽기
        text_content = ""
        if os.path.exists(text_filepath[0]):
            with open(text_filepath[0], 'r') as text_file:
                text_content = text_file.read()
        else:
            return jsonify({'error': 'Text file not found'}), 404

        # JSON 응답으로 이미지(Base64)와 텍스트 내용 전송
        response = {
            'image': encoded_image,  # Base64 인코딩된 처리된 이미지 데이터
            'text': text_content     # 텍스트 파일의 내용
        }

        return jsonify(response), 200

    return jsonify({'error': 'Invalid file format'}), 400


if __name__ == '__main__':
    app.run(debug=True)
