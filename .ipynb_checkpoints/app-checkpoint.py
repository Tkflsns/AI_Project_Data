from flask import Flask, request, jsonify
from image_processing_pipeline import process_image  # process_image 함수 사용
import os
import base64
from waitress import serve

app = Flask(__name__)

# 최대 업로드 크기 설정 (1000MB)
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 1000MB

UPLOAD_FOLDER = 'input/'
OUTPUT_FOLDER = 'output/'
PROCESSED_FOLDER = 'Boxed_crop/gray'  # 처리된 이미지가 저장되는 폴더 경로

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, PROCESSED_FOLDER), exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 파일 업로드 및 처리 후 이미지와 텍스트 전송
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 업로드되지 않았습니다.'}), 400

    # 파일 목록 가져오기 (단일 파일일 경우에도 리스트로 처리)
    files = request.files.getlist('file')

    if not files:
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400

    results = []
    total_score = 0.0
    processed_count = 0

    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            input_filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_filepath)

            try:
                # 이미지 처리 후 결과 텍스트 파일 경로 반환
                image_file, text_filepath, confidence_scores = process_image(input_filepath)

                # 처리된 이미지 파일 경로 설정 (Boxed_crop/gray 디렉토리에서 이름이 같은 파일)
                processed_image_filename = os.path.basename(image_file)
                processed_image_path = os.path.join(OUTPUT_FOLDER, PROCESSED_FOLDER, processed_image_filename)

                # 이미지 파일을 Base64로 인코딩
                if os.path.exists(processed_image_path):
                    with open(processed_image_path, "rb") as image_file_obj:
                        encoded_image = base64.b64encode(image_file_obj.read()).decode('utf-8')
                else:
                    results.append({'filename': filename, 'error': '처리된 이미지를 찾을 수 없습니다.'})
                    continue

                # 텍스트 파일 내용을 읽기
                text_content = ""
                if os.path.exists(text_filepath[0]):
                    with open(text_filepath[0], 'r') as text_file:
                        text_content = text_file.read()
                        text_content = "".join(text_content.split())
                else:
                    results.append({'filename': filename, 'error': '텍스트 파일을 찾을 수 없습니다.'})
                    continue

                # 신뢰도 점수 누적
                score_cumprod = float(confidence_scores[0])
                score_mean = float(confidence_scores[1])

                total_score_cumprod += score_cumprod
                total_score_mean += score_mean
                processed_count += 1

                # 결과 추가
                results.append({
                    'filename': filename,
                    'image': encoded_image,
                    'text': text_content,
                    'score_cumprod': f'{round(score_cumprod*100, 1)}%',
                    'score_mean': f'{round(score_mean*100, 1)}%'
                })

            except Exception as e:
                results.append({'filename': filename, 'error': str(e)})

        else:
            results.append({'filename': file.filename, 'error': '유효하지 않은 파일 형식입니다.'})

    # 평균 점수 계산
    average_score_cumprod = 0.0
    average_score_mean = 0.0
    if processed_count > 0:
        average_score_cumprod = total_score_cumprod / processed_count
        average_score_mean = total_score_mean / processed_count

    return jsonify({
        'results': results,
        'average_score_cumprod': f'{round(average_score_cumprod*100, 1)}%',
        'average_score_mean': f'{round(average_score_mean*100, 1)}%'
    }), 200

# 오류 처리: 파일 크기 초과 시 응답
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': '업로드된 파일이 너무 큽니다. 최대 허용 크기는 1000MB입니다.'}), 413

if __name__ == '__main__':
    # Flask 개발 서버 대신 Waitress로 애플리케이션 실행
    serve(app, host='0.0.0.0', port=8080)
