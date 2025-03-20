# 설정 관련 기능들을 묶어 분할한다.
import os
import json
import traceback

from PyQt5.QtWidgets import QFileDialog, QVBoxLayout, QWidget, QDialog
from core.dialog.setting_dialog import SettingsDialog

class SettingsHandler:
    def __init__(self, parent, config_file):
        self.parent = parent
        
        # 사용자 홈 디렉토리에 .imagekeywordextractor 폴더 생성
        self.app_dir = os.path.join(os.path.expanduser("~"), ".imagekeywordextractor")
        if not os.path.exists(self.app_dir):
            os.makedirs(self.app_dir, exist_ok=True)
        
        # config_file 경로를 app_dir 내부로 설정
        self.config_file = os.path.join(self.app_dir, os.path.basename(config_file))
        
        print(f"설정 파일 경로: {self.config_file}")
        
        # 기본 설정값 가져오기
        self.settings = {
            'openai_key': '',
            'last_save_directory': os.path.expanduser("~"),  # 홈 디렉토리로 설정
            'assistant_id': ''  # 빈 문자열로 변경
        }
        
        # config.json이 없으면 기본값으로 생성
        if not os.path.exists(self.config_file):
            # 직접 파일 생성
            try:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.settings, f, ensure_ascii=False, indent=4)
                print(f"설정 파일 생성됨: {self.config_file}")
            except Exception as e:
                print(f"설정 파일 생성 오류: {str(e)}")
        else:
            try:
                with open(self.config_file, 'r') as f:
                    saved_settings = json.load(f)
                    # 기존 설정에 누락된 키가 있으면 기본값 추가
                    for key, value in self.settings.items():
                        if key not in saved_settings:
                            saved_settings[key] = value
                    self.settings = saved_settings
            except:
                # 파일이 손상된 경우 기본값으로 새로 생성
                try:
                    with open(self.config_file, 'w', encoding='utf-8') as f:
                        json.dump(self.settings, f, ensure_ascii=False, indent=4)
                    print(f"손상된 설정 파일 재생성됨: {self.config_file}")
                except Exception as e:
                    print(f"설정 파일 재생성 오류: {str(e)}")

    def load_settings(self):
        """설정 파일에서 설정 불러오기"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                print(f"설정 파일 불러옴: {self.config_file}")
                
                # 기존 설정 업데이트
                self.settings.update(loaded_config)
                
                # 디렉토리 경로 유효성 확인
                for key in ['load_dir', 'last_save_directory']:
                    if key in self.settings and self.settings[key]:
                        if not os.path.exists(self.settings[key]):
                            print(f"경고: 설정된 디렉토리가 존재하지 않음 - {key}: {self.settings[key]}")
                
                return True
            else:
                print(f"설정 파일이 존재하지 않음: {self.config_file}")
                return False
        except Exception as e:
            print(f"설정 불러오기 오류: {str(e)}")
            return False

    def check_settings(self):
        if not os.path.exists(self.config_file):
            return False
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        return bool(config.get('openai_key'))

    def save_settings(self, settings_to_save=None):
        """설정 저장 메서드 수정"""
        try:
            # 현재 설정과 새 설정 병합
            if settings_to_save:
                self.settings.update(settings_to_save)
            
            # 설정 파일에 저장
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=4)
            
            return True
        except Exception as e:
            print(f"설정 저장 중 오류 발생: {str(e)}")
            return False

    def save_setting(self, key, value):
        """단일 설정값 저장"""
        old_value = self.settings.get(key)
        if old_value != value:  # 값이 변경된 경우에만 저장
            self.settings[key] = value
            print(f"설정 변경: {key} = {value}")
            return self.save_settings()
        return True

    def get_setting(self, key, default=None):
        """설정값 가져오기"""
        value = self.settings.get(key, default)
        # 디렉토리 경로인 경우 유효성 확인
        if key in ['load_dir', 'last_save_directory'] and value:
            if not os.path.exists(value):
                print(f"경고: 요청한 디렉토리 경로가 존재하지 않음 - {key}: {value}")
                # 사용자 홈 디렉토리로 대체
                if default is None:
                    return os.path.expanduser('~')
        return value

    def open_settings_dialog(self):
        settings_dialog = SettingsDialog(self.parent)
        result = settings_dialog.exec_()
        if result == QDialog.Accepted:
            # 설정이 저장된 후 즉시 새로운 설정을 로드하고 반환
            api_key = self.load_settings()
            if api_key:
                print(f"Loaded API key after saving: {bool(api_key)}")  # 디버깅용
                return result
        return result

    def get_default_settings(self):
        """기본 설정 반환"""
        return {
            "openai_key": "",
            "assistant_id": "",  # 비워두기
            "auto_save": False,
            "last_save_directory": os.path.expanduser("~")
        }
