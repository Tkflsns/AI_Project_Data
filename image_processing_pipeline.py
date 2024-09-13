import torch
import gc
import os
import shutil
from Number_P_Result import NP_Detect
from Realesrgan_upscale import enhance_image
from text_search import process_text

# 각 파일에서 호출할 함수 정의
def detect_license_plate(image_file):
    filename = image_file  # 실제 파일 경로 입력
    detected_image_path = NP_Detect(filename)  # 번호판 검출 후 반환되는 경로
    return detected_image_path  # 번호판 검출된 이미지 경로 반환

def upscale_image(image_file):
    input_file = image_file  # 실제 파일 경로 입력
    output_folder = 'output/Upscaled/'
    
    # 업스케일 결과 파일 경로 반환
    enhanced_image_path = enhance_image(input_file, output_folder)
    print(f'Enhanced image saved at: {enhanced_image_path}')
    return enhanced_image_path  # 업스케일된 이미지 경로 반환

def extract_text(image_file):
    # 예시: 외부에서 파일 경로를 받아 처리
    image_file = image_file  # 실제 이미지 파일 경로
    output_folder = 'output/Number_Search/'  # 실제 출력 폴더 경로
    
    # 텍스트 처리 결과 파일 경로 반환
    result_file_paths, confidence_scores = process_text(image_file, output_folder)
    print(f"Result files saved at: {result_file_paths}")
    
    return result_file_paths, confidence_scores  # 추출된 텍스트 파일 경로 반환

def process_image(image_file):
    # 1. 번호판 검출
    plate_image_file = detect_license_plate(image_file)
    
    # 2. 화질 업스케일
    upscaled_image_file = upscale_image(plate_image_file)
    
    # 3. 텍스트 추출
    text, confidence_scores = extract_text(upscaled_image_file)

    # 메모리 해제
    gc.collect()
    torch.cuda.empty_cache()
    
    return plate_image_file, text, confidence_scores

if __name__ == "__main__":
    # 테스트용 이미지 파일 경로
    test_image_filename = 'image.jpg'
    
    # 이미지 처리 실행 및 텍스트 추출
    extracted_text = process_image(test_image_filename)
    print(f"Extracted Text: {extracted_text}")
