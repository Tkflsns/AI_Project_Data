import os
import string
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from lincenseplateocr.utils import CTCLabelConverter, AttnLabelConverter
from lincenseplateocr.model import Model
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def demo(opt):
    """ Model configuration """
    print("Initializing converter...")  # 디버깅 추가
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    print(f"Number of classes: {opt.num_class}")

    print("Setting input channels...")  # 디버깅 추가
    # 모델 입력 채널 설정
    if opt.rgb:
        opt.input_channel = 3  # RGB 이미지
    else:
        opt.input_channel = 1  # 그레이스케일 이미지

    print("Initializing model...")  # 디버깅 추가
    model = Model(opt)
    print("Model initialized.")  # 모델이 정상적으로 초기화되었는지 확인

    # 데이터 병렬처리 설정 (문제가 발생할 수 있으므로 테스트할 필요가 있음)
    print("Setting model to DataParallel...")
    model = torch.nn.DataParallel(model).to(device)
    print("Model set to DataParallel.")

    # 모델 가중치 로드
    print(f"Loading pretrained model from {opt.saved_model}...")
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    print("Model loaded.")

    # 이미지 전처리 준비 (1채널로 변환)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 그레이스케일로 변환 추가
        transforms.Resize((opt.imgH, opt.imgW)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 단일 이미지 파일 처리
    image = Image.open(opt.image_folder).convert('RGB')
    image.show()
    image = transform(image).unsqueeze(0).to(device)  # 배치 차원을 추가하고 GPU로 보냄

    model.eval()
    output_file_paths = []
    with torch.no_grad():
        length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
        text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)])
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
        else:
            preds = model(image, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        img_name = os.path.basename(opt.image_folder)
        pred = preds_str[0]
        pred_max_prob = preds_max_prob[0]

        if 'Attn' in opt.Prediction:
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]
            pred_max_prob = pred_max_prob[:pred_EOS]

        confidence_score = pred_max_prob.cumprod(dim=0)[-1]

        # output 폴더가 없으면 생성
        if not os.path.exists(opt.output_folder):
            os.makedirs(opt.output_folder)

        # 결과 저장 경로
        log_path = os.path.join(opt.output_folder, f'{os.path.splitext(img_name)[0]}.txt')
        with open(log_path, 'w') as log:
            log.write(f'{pred:25s}')

        output_file_paths.append(log_path)

    return output_file_paths

# Jupyter 환경에서는 명령줄 인자를 수동으로 설정합니다.
class Opt:
    def __init__(self, image_file, output_folder):
        self.image_folder = image_file  # 단일 이미지 파일 경로
        self.output_folder = output_folder  # 외부에서 입력받은 출력 폴더 경로
        self.workers = 0
        self.batch_size = 1
        self.saved_model = 'lincenseplateocr/pretrained/iter_50000.pth'
        self.batch_max_length = 16
        self.imgH = 32
        self.imgW = 100
        self.rgb = False  # 1채널 (그레이스케일)로 설정
        self.character = '0123456789().JNRW_abcdef가강개걍거겅겨견결경계고과관광굥구금기김깅나남너노논누니다대댜더뎡도동두등디라러로루룰리마머명모무문므미바배뱌버베보부북비사산서성세셔소송수시아악안양어여연영오올용우울원육으을이익인자작저전제조종주중지차처천초추출충층카콜타파평포하허호홀후히ㅣ'
        self.sensitive = False
        self.PAD = False
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        self.num_fiducial = 20
        self.input_channel = 1  # 그레이스케일 이미지를 입력으로 받음
        self.output_channel = 512
        self.hidden_size = 256

# 직접 설정한 옵션 객체 생성
def process_text(image_file, output_folder):
    opt = Opt(image_file, output_folder)

    if opt.sensitive:
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    # 모델 실행 및 결과 텍스트 파일 경로 리턴
    output_file_paths = demo(opt)
    return output_file_paths
