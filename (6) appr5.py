from flask import Flask, render_template, Response, jsonify, request
import numpy as np
import ssl
import os
import cv2
import dlib
import numpy as np
import time
import datetime
from datetime import datetime, timedelta
import math
from math import hypot
from scipy.spatial import distance
from collections import Counter
from scipy.spatial import distance as dist
import logging
from logging.handlers import RotatingFileHandler

# import sys

def setup_logging(app):
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/myapp.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))

    file_handler.setLevel(logging.INFO)  # 또는 필요에 따라 DEBUG
    app.logger.addHandler(file_handler)

    app.logger.setLevel(logging.INFO)  # 또는 필요에 따라 DEBUG
    app.logger.info('MyApp startup')

app = Flask(__name__)
setup_logging(app)
context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain("/home/ubuntu/myflaskapp/src/ssl.crt", "/home/ubuntu/myflaskapp/src/ssl.key")

### Dlib 얼굴 검출기 초기화 ************************************
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


# 눈 깜박임 비율을 저장할 리스트
blink_ratios_history = []
blink_ratio = None


# 깜박임 감지를 위한 임계값 설정
BLINK_RATIO_THRESHOLD = 4.8 # 이 값은 실험을 통해 적절한 값을 찾아야 합니다.
#CLOSED_EYES_FRAME_THRESHOLD = 9  # 눈을 감은 것으로 간주할 프레임 수
CLOSED_EYES_FRAME_THRESHOLD = 0.5  # 초
start_time = 0
closed_eyes_time = 0


# 상수들 정의
EYE_AR_THRESH = 0.3  # 눈 깜빡임 감지 임계값
EYE_AR_CONSEC_FRAMES = 3  # 눈을 감고 있는 프레임의 연속 수
MOUSE_MOVEMENT_SCALE = 2.0  # 마우스 움직임의 척도

EYE_POINTS_LEFT = [36, 37, 38, 39, 40, 41]  # 왼쪽 눈의 landmark 점들
EYE_POINTS_RIGHT = [42, 43, 44, 45, 46, 47]  # 오른쪽 눈의 landmark 점들

#BLINK_RATIO_THRESHOLD = 5.7, 4.8이맞음.


# 눈을 감은 프레임을 추적하기 위한 변수
closed_eyes_frame_counter = 0


# 졸림 상태 및 프레임 카운터를 추적하기 위한 변수
drowsy_alert_active = False
drowsy_frame_counter = 0

# 랜드마크의 중간점을 계산하는 함수
def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def show_alert():
    """
    사용자에게 팝업 알림을 보여주는 함수입니다.
    """
    # Tk 객체 인스턴스화. 일반적으로 Tkinter 애플리케이션의 첫 번째 단계입니다.
    root = tk.Tk()
    root.withdraw()  # 기본적으로 생성되는 빈 창을 숨깁니다.

    # 팝업 메시지
    messagebox.showwarning("경고!", "졸음을 감지했습니다! 안전을 위해 휴식을 취하십시오!")

    # 이벤트 루프 종료 후 창 닫기
    root.destroy()
    

# 시선 비율을 계산하는 함수
def get_gaze_ratio(eye_points, facial_landmarks, frame, gray):
    # 눈 영역의 좌표를 구합니다.
    eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points])

    # 눈 영역에서 최소/최대 x 및 y 좌표를 찾아 눈 영역을 만듭니다.
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    # 그레이스케일 이미지에서 눈 영역을 추출합니다.
    eye = gray[min_y: max_y, min_x: max_x]

    # 눈 영역의 너비와 높이가 충분히 큰지 확인합니다.
    if eye.size == 0 or eye.shape[0] < 2 or eye.shape[1] < 2:
        return None

    # 눈동자 감지를 위해 threshold를 적용합니다.
    _, eye = cv2.threshold(eye, 70, 255, cv2.THRESH_BINARY_INV)
    
    # 눈동자의 위치를 찾기 위한 코드 (예: 중심 찾기, 가중치 적용 등)
    # 이 부분은 실제 눈동자 위치를 정확하게 찾는 로직으로 변경되어야 합니다.
    # 예시로, 단순하게 흰색 픽셀(눈동자)의 수를 계산하는 것을 사용할 수 있습니다.
    white_pixels = cv2.countNonZero(eye)

    # 눈 영역의 총 픽셀 수를 계산합니다.
    total_pixels = eye.shape[0] * eye.shape[1]

    # 감지된 흰색 픽셀의 비율을 계산합니다.
    gaze_ratio = white_pixels / total_pixels if total_pixels > 0 else 0

    return gaze_ratio
    

def detect_faces(frame):
    try:
        # 프레임의 차원과 채널 확인
        if frame is None:
            print("No frame available")
            return None, None

        if len(frame.shape) == 2:
            # 이미 그레이스케일
            gray = frame
        elif len(frame.shape) == 3:
            # 컬러 이미지, 그레이스케일로 변환 필요
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # 예상치 못한 경우, 오류 로깅
            print("Unexpected number of image channels:", frame.shape)
            return None, None

        # 얼굴 감지 수행 (detector는 미리 정의된 dlib의 face detector 등이 될 수 있습니다.)
        faces = detector(gray)
        return faces, gray

    except cv2.error as e:
        # OpenCV 관련 예외 처리
        print(f"An OpenCV error occurred: {e}")
    except Exception as e:
        # 기타 예외 처리
        print(f"An error occurred: {e}")

    # 예외가 발생한 경우 None 반환
    return None, None


# 주요 기능을 수행하는 함수: 얼굴 감지, 랜드마크 추출, 비율 계산
def draw_faces(frame, faces, gray):
    blink_ratios = []
    gaze_ratios = []
    nose_ratios = []

    for face in faces:
        landmarks = predictor(gray, face)

        # 깜박임 감지 부분의 코드
        left_point = (landmarks.part(36).x, landmarks.part(36).y)  # .part 메서드 사용
        right_point = (landmarks.part(39).x, landmarks.part(39).y)  # .part 메서드 사용
        center_top =  midpoint(landmarks.part(37), landmarks.part(38)) # .part 메서드 사용
        center_bottom = midpoint(landmarks.part(41), landmarks.part(40)) # .part 메서드 사용
        
        
        # 비율 계산
        hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        if ver_line_length > 0:  # 0으로 나누는 것을 방지
            ratio = hor_line_length / ver_line_length
            blink_ratios.append(ratio)

        # 시선 감지 부분의 코드
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, frame, gray)  # 매개변수 추가
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, frame, gray)  # 매개변수 추가
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2
        gaze_ratios.append(gaze_ratio)

        ### 머리 회전 반경 계산
        end_nose_point = (landmarks.part(29).x, landmarks.part(29).y)
        left_libs_point = (landmarks.part(4).x, landmarks.part(4).y)
        right_libs_point = (landmarks.part(12).x, landmarks.part(12).y)

        #코와 입의 길이 계산
        nose_line_len_left = hypot(left_libs_point[0]-end_nose_point[0],left_libs_point[1]-end_nose_point[1])
        nose_line_len_right = hypot(right_libs_point[0]-end_nose_point[0],right_libs_point[1]-end_nose_point[1])
        nose_ratio = nose_line_len_left/nose_line_len_right 
        nose_ratios.append(nose_ratio)

    return blink_ratios, gaze_ratios, nose_ratios

def calculate_ratios(landmarks):
    # 눈 깜빡임 비율 계산
    left_point = (landmarks.part(36).x, landmarks.part(36).y)
    right_point = (landmarks.part(39).x, landmarks.part(39).y)
    center_top = midpoint(landmarks.part(37), landmarks.part(38))
    center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

    hor_line_len = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_len = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    blink_ratio = hor_line_len / ver_line_len if ver_line_len > 0 else hor_line_len

    # Here you can add calculations for other ratios like gaze_ratio, nose_ratio, etc.

    return blink_ratio  # You can return more ratios as needed

@app.route('/')
def index():
    return render_template('/video/main.html')

### 영상 구동 페이진
@app.route('/video_feed')
def video_feed():
    return render_template('/video/video.html')


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/process', methods=['POST'])
def process():
    global closed_eyes_time, start_time

    try:
        # 클라이언트로부터 이미지 데이터 수신
        blob = request.data
        nparr = np.frombuffer(blob, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            return jsonify(status="error", message="Empty or corrupted image received")

        # 얼굴 및 랜드마크 검출
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, gray = detect_faces(gray)

        if faces is None:
            return jsonify(status="error", message="No faces detected")

        # 감지된 각 얼굴에 대해 처리
        drowsy_alert_active = False
        for face in faces:
            # 'face'가 dlib.rectangle 인스턴스인지 확인
            if isinstance(face, dlib.rectangle):
                landmarks = predictor(gray, face)
                blink_ratio = calculate_ratios(landmarks)

                if blink_ratio > BLINK_RATIO_THRESHOLD:
                    if start_time == 0:
                        start_time = time.time()  # 눈을 감기 시작한 시간 기록
                    else:
                        closed_eyes_time = time.time() - start_time

                        if closed_eyes_time > CLOSED_EYES_FRAME_THRESHOLD:
                            drowsy_alert_active = True
                else:
                    start_time = 0

        response_data = {
            "status": "success",
            "blink_ratio": blink_ratio,
            "closed_eyes_time": closed_eyes_time,
            "drowsy_alert_active": drowsy_alert_active,
        }

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"An error occurred during processing: {e}", exc_info=True)
        return jsonify(status="error", message=str(e))

if __name__ == '__main__':
    # 환경 변수에서 SSL 키 비밀번호 가져오기
    ssl_password = os.environ.get('SSL_KEY_PASSWORD')

     # SSL 컨텍스트 설정
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain("/home/ubuntu/myflaskapp/src/ssl.crt", "/home/ubuntu/myflaskapp/src/ssl.key")
    # app.run(debug=True)
    
    # debug=True는 에러가 없으면, 자동으로 서버 재시작함
    # 코드 수정 시 애러가 없으면, 서버가 재 실행 됨
    app.debug = True 
    
    # run에는 여러가지 옵션이 있음
    # app.run(host="0.0.0.0", port="5000", debug=True)
   # Flask 애플리케이션 실행
    app.run(host='0.0.0.0', port="5000", ssl_context=context,threaded=True, debug=True)