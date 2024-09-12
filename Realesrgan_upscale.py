import cv2
import os
import re
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def remove_korean(text):
    # 한글을 삭제한 파일 이름을 반환
    return re.sub(r'[가-힣]', '', text)

def temp_rename_file_remove_korean(file_path):
    # 파일 이름에서 한글을 제거하고, 원래 이름과 매핑된 딕셔너리 반환
    filename = os.path.basename(file_path)
    english_name = remove_korean(filename)
    new_path = os.path.join(os.path.dirname(file_path), english_name)
    os.rename(file_path, new_path)
    return new_path, file_path  # 변경된 이름과 원래 이름 매핑

def revert_filename(new_name, original_name):
    # 변경된 이름을 원래 이름으로 복원
    os.rename(new_name, original_name)

def enhance_image(input_file, output_folder, model_name='RealESRNet_x4plus', denoise_strength=0.5, outscale=4, 
                   model_path='./weights/RealESRNet_x4plus.pth', suffix='', tile=0, tile_pad=10, pre_pad=0, face_enhance=False, 
                   fp32=True, alpha_upsampler='realesrgan', ext='auto', gpu_id=None):
    
    # 한글을 제거한 임시 파일 이름으로 변경
    new_file_path, original_file_path = temp_rename_file_remove_korean(input_file)

    try:
        # 모델 선택에 따른 설정
        model_name = model_name.split('.')[0]
        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet 모델
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet 모델
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet 모델 (6블록)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet 모델
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif model_name == 'realesr-animevideov3':  # x4 VGG 스타일 모델 (XS 크기)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif model_name == 'realesr-general-x4v3':  # x4 VGG 스타일 모델 (S 크기)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]

        # 모델 경로 결정
        if model_path is not None:
            model_path = model_path
        else:
            model_path = os.path.join('weights', model_name + '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(''))
                for url in file_url:
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

        # 노이즈 제거 강도를 제어하는 DNI 사용
        dni_weight = None
        if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [denoise_strength, 1 - denoise_strength]

        # 복원기 설정
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp32,
            gpu_id=gpu_id)

        os.makedirs(output_folder, exist_ok=True)

        # 이미지 처리
        imgname, extension = os.path.splitext(os.path.basename(original_file_path))  # 원래 파일 이름을 사용
        print('Processing', imgname)

        img = cv2.imread(new_file_path, cv2.IMREAD_UNCHANGED)
            
        if img is None:
            print(f"Error reading image {new_file_path}, skipping...")
            return None

        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            output, _ = upsampler.enhance(img, outscale=outscale)
        except RuntimeError as error:
            print('Error:', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            return None

        # 확장자 처리
        if ext == 'auto':
            extension = extension[1:]
        else:
            extension = ext
        if img_mode == 'RGBA':  # RGBA 이미지들은 PNG 형식으로 저장되어야 함
            extension = 'png'
        if suffix == '':
            save_path = os.path.join(output_folder, f'{imgname}.{extension}')
        else:
            save_path = os.path.join(output_folder, f'{imgname}_{suffix}.{extension}')
        cv2.imwrite(save_path, output)
        print(f"Saved: {save_path}")
        return save_path

    finally:
        # 작업 후 파일 이름 복원
        revert_filename(new_file_path, original_file_path)

# 처리된 파일 경로가 원래 파일 이름을 기반으로 저장되도록 설정하였고, 복원된 경로를 리턴합니다.


# enhance_images(
#     input_path= os.path.abspath('data/img_labeling/dh/cropped_number_plate_gray/'),
#     output_path= os.path.abspath('data/img_labeling/dh/cropped_number_plate_gray/up/'),
#     model_name='RealESRGAN_x4plus_anime_6B',
#     denoise_strength=1,
#     outscale=4,
#     model_path= None,
#     suffix='out',
#     tile=0,
#     tile_pad=10,
#     pre_pad=0,
#     face_enhance=False,
#     fp32=True,
#     alpha_upsampler='realesrgan',
#     ext='auto',
#     gpu_id=None
# )

# enhance_images(
#     input_path= os.path.abspath('data/img_labeling/dh/cropped_number_plate/upscaled1/'),
#     output_path= os.path.abspath('data/img_labeling/dh/cropped_number_plate/upscaled2/'),
#     model_name='RealESRGAN_x4plus_anime_6B',
#     denoise_strength=0.8,
#     outscale=4,
#     model_path= None,
#     suffix='out2',
#     tile=0,
#     tile_pad=10,
#     pre_pad=0,
#     face_enhance=False,
#     fp32=True,
#     alpha_upsampler='realesrgan',
#     ext='auto',
#     gpu_id=None
# )
# 설명: 입력 이미지 파일 또는 폴더의 경로를 지정합니다. 만약 폴더 경로를 지정하면 해당 폴더 내의 모든 이미지가 처리됩니다.
# 예시: 'inputs' (폴더 경로) 또는 'inputs/image.jpg' (파일 경로).
# output_path:

# 설명: 처리된 이미지를 저장할 출력 폴더의 경로를 지정합니다. 이 폴더는 존재하지 않으면 자동으로 생성됩니다.
# 예시: 'results'.
# model_name:

# 설명: 사용할 Real-ESRGAN 모델의 이름을 지정합니다. 아래의 모델 중 하나를 선택할 수 있습니다:
# 'RealESRGAN_x4plus': 일반적인 이미지 슈퍼 해상도를 위한 모델.
# 'RealESRNet_x4plus': 노이즈가 줄어든 버전의 모델.
# 'RealESRGAN_x4plus_anime_6B': 애니메이션 이미지에 특화된 모델.
# 'RealESRGAN_x2plus': 2배 슈퍼 해상도를 위한 모델.
# 'realesr-animevideov3': 애니메이션 비디오에 특화된 모델.
# 'realesr-general-x4v3': 다양한 이미지에 사용할 수 있는 범용 모델.
# 예시: 'RealESRGAN_x4plus'.
# denoise_strength:

# 설명: 노이즈 제거 강도를 설정합니다. 값이 0이면 약한 노이즈 제거, 1이면 강한 노이즈 제거를 의미합니다. 이 파라미터는 'realesr-general-x4v3' 모델에서만 사용됩니다.
# 예시: 0.5.
# outscale:

# 설명: 최종 업스케일 배율을 설정합니다. 예를 들어, 값이 4이면 이미지의 해상도가 4배로 증가합니다.
# 예시: 4.
# model_path:

# 설명: 사전 학습된 모델 파일의 경로를 지정합니다. 지정하지 않으면 코드가 자동으로 모델을 다운로드하여 사용합니다.
# 예시: None (자동 다운로드) 또는 'path/to/model.pth'.
# suffix:

# 설명: 출력 이미지 파일명에 추가될 접미사를 지정합니다. 예를 들어, 'out'으로 지정하면, 결과 파일명은 'image_out.png'가 됩니다.
# 예시: 'out'.
# tile:

# 설명: 타일 크기를 지정합니다. 타일링은 메모리 사용을 줄이기 위해 이미지의 일부를 나누어 처리하는 방법입니다. 값이 0이면 타일 없이 처리합니다.
# 예시: 0 (타일링 비활성화) 또는 512 (512x512 크기의 타일 사용).
# tile_pad:

# 설명: 타일 간의 겹침 영역(패딩)을 설정합니다. 타일링을 사용할 때 경계 부분의 연속성을 보장하기 위해 사용됩니다.
# 예시: 10.
# pre_pad:

# 설명: 이미지 처리 전 가장자리 패딩 크기를 설정합니다. 값이 0이면 패딩이 없습니다.
# 예시: 0.
# face_enhance:

# 설명: 얼굴 향상을 위해 GFPGAN 모델을 사용할지 여부를 설정합니다. 이 옵션이 활성화되면 얼굴 부분이 더 정밀하게 복원됩니다.
# 예시: False (비활성화) 또는 True (활성화).
# fp32:

# 설명: FP32(32-bit floating point) 정밀도를 사용할지 설정합니다. 기본값은 FP16(half precision)으로, 메모리 사용량을 줄이고 속도를 높일 수 있습니다.
# 예시: False (FP16 사용) 또는 True (FP32 사용).
# alpha_upsampler:

# 설명: 알파 채널(투명도 채널) 업샘플링에 사용할 방식을 지정합니다. 선택 가능한 옵션은 'realesrgan' 또는 'bicubic'입니다.
# 예시: 'realesrgan' (Real-ESRGAN 방식 사용) 또는 'bicubic' (Bi-cubic 방식 사용).
# ext:

# 설명: 출력 이미지의 파일 확장자를 지정합니다. 'auto'로 설정하면 입력 이미지와 동일한 확장자를 사용합니다.
# 예시: 'auto', 'jpg', 'png'.
# gpu_id:

# 설명: 사용할 GPU의 ID를 지정합니다. 다중 GPU 환경에서 특정 GPU를 선택할 수 있습니다. 기본값은 None으로, 이 경우 기본 GPU가 사용됩니다.
# 예시: 0 (첫 번째 GPU 사용), 1 (두 번째 GPU 사용) 또는 None (기본 설정 사용).




