import os
import sys
from pathlib import Path

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# 사용자 홈 디렉토리에 .imagekeywordextractor 폴더 생성 및 config.json 경로 설정
def get_config_path():
    """사용자 홈 디렉토리의 .imagekeywordextractor/config.json 경로 반환"""
    config_dir = os.path.join(os.path.expanduser("~"), ".imagekeywordextractor")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "config.json")

base_dir = resource_path(".")
cfg_path = get_config_path()  # 홈 디렉토리의 config.json 경로 사용
default_load_dir = str(Path.home() / "Documents")
default_save_dir = str(Path.home() / "Documents") #document(문서)로 바로 갈 수 있게 수정하기.
img_ext = ["jpg", "jpeg", "png", "bmp"]
# ui_dir = "./res/ui/" #기본 디렉토리 사용해서 다른 코드들 수정하기
ui_dir = resource_path("res/ui")
css_dir = resource_path("res/css")