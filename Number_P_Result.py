import os
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
from tqdm import tqdm
from ultralytics import YOLO

def NP_Detect(filepath):

    # 학습된 YOLO 모델 로드
    model = YOLO('weights/plate_detect.pt')  # 학습된 모델 경로
    
    # 이미지가 있는 폴더 경로 설정 (입력 이미지와 동일 경로에 처리 결과를 저장하도록 수정)
    input_image_path = os.path.abspath(filepath)  # 입력으로 받은 단일 이미지 파일
    output_folder = os.path.abspath('output/Boxed_crop/')  # 결과 이미지를 저장할 폴더
    cropped_output_folder = os.path.join(output_folder, 'gray/')  # 크롭된 번호판 이미지를 저장할 폴더
    labels_folder = os.path.join(cropped_output_folder, 'labels/')  # YOLO 형식 라벨을 저장할 폴더
    
    # 결과 이미지를 저장할 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(cropped_output_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # 이미지 파일 처리
    if filepath.endswith('.jpg') or filepath.endswith('.png'):  # 지원하는 이미지 형식
        # 이미지 경로 설정
        output_image_path = os.path.join(output_folder, os.path.basename(filepath))
        label_file_path = os.path.join(labels_folder, os.path.basename(filepath).replace('.jpg', '.txt'))
    
        # 이미지 불러오기 (PIL 사용)
        pil_image = Image.open(input_image_path)
        width, height = pil_image.size  # PIL 이미지의 크기
    
        # 모델을 사용해 번호판 객체 감지
        results = model(pil_image)
    
        # 라벨 파일 쓰기 위해 준비
        label_lines = []
    
        # 감지된 바운딩 박스 좌표 및 클래스 정보 출력 및 그리기
        draw = ImageDraw.Draw(pil_image)  # PIL의 이미지에 그리기 객체 생성
    
        for idx, box in enumerate(results[0].boxes):
            # 바운딩 박스 좌표 및 클래스 정보
            bbox = box.xyxy[0].cpu().numpy()  # 바운딩 박스 좌표 (x_min, y_min, x_max, y_max)
            class_id = int(box.cls[0].cpu().numpy())  # 클래스 ID
            confidence = box.conf[0].cpu().numpy()  # 신뢰도
    
            # 클래스 ID가 번호판인지 확인 (여기서는 클래스 ID 0번이 번호판이라고 가정)
            if class_id == 0:  # 번호판 클래스 ID
                # 바운딩 박스 좌표 (정수로 변환)
                x_min, y_min, x_max, y_max = map(int, bbox)
                    
                # 바운딩 박스 그리기 (파란색, 두께 2)
                draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=2)
    
                # 신뢰도 및 클래스 ID 표시
                label = f"Number Plate {confidence:.2f}"
                draw.text((x_min, y_min - 10), label, fill="blue")
    
                # 크롭된 이미지 저장
                cropped_image = pil_image.crop((x_min, y_min, x_max, y_max))
    
                # 이미지를 회색조로 변환
                gray_image = ImageOps.grayscale(cropped_image)
    
                # 대비 조정 (1.0은 기본값, 2.0은 대비를 2배로 증가)
                enhancer = ImageEnhance.Contrast(gray_image)
                contrast_image = enhancer.enhance(1.0)
    
                cropped_image_path = os.path.join(cropped_output_folder, f"{os.path.splitext(os.path.basename(filepath))[0]}.jpg")
                contrast_image.save(cropped_image_path)
    
                # YOLO 형식의 라벨 작성
                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                bbox_width = (x_max - x_min) / width
                bbox_height = (y_max - y_min) / height
                label_lines.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")
    
        # 라벨 파일 저장 (YOLO 형식)
        with open(label_file_path, 'w') as label_file:
            label_file.writelines(label_lines)
    
        # 결과 이미지 저장 (전체 이미지)
        pil_image.save(output_image_path)

    print(f"객체 추적 및 박싱 완료. 결과 이미지는 {output_folder}, {cropped_output_folder}, {labels_folder} 폴더에 저장되었습니다.")
    return cropped_image_path
