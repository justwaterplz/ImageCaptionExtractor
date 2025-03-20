import sys
import os
import json
from PyQt5.QtWidgets import QApplication
from core.dialog.main_dialog import MainUI
from core.dialog.setting_dialog import SettingsDialog
from cfg.cfg import cfg_path

def check_settings_file():
    """설정 파일 존재 여부 확인 및 생성"""
    # cfg_path는 이제 홈 디렉토리의 .imagekeywordextractor/config.json을 가리킴
    print(f"설정 파일 경로: {cfg_path}")
    
    if not os.path.exists(cfg_path):
        print(f"Settings file not found at {cfg_path}")
        # 설정 파일이 저장될 디렉토리 확인
        config_dir = os.path.dirname(cfg_path)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
            print(f"Created config directory: {config_dir}")
            
        # 기본 설정으로 파일 생성
        default_settings = {
            'openai_key': '',
            'load_dir': os.path.expanduser('~'),
            'last_save_directory': os.path.expanduser('~'),
            'assistant_id': ''
        }
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(default_settings, f, ensure_ascii=False, indent=4)
        print(f"Created new settings file at {cfg_path}")
        return False
    
    # 설정 파일이 있는 경우, 모든 필수 키가 있는지 확인하고 업데이트
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        is_updated = False
        
        # 필수 키 목록
        required_keys = {
            'openai_key': '',
            'load_dir': os.path.expanduser('~'),
            'last_save_directory': os.path.expanduser('~'),
            'assistant_id': ''
        }
        
        # 누락된 키가 있으면 기본값으로 추가
        for key, default_value in required_keys.items():
            if key not in settings:
                settings[key] = default_value
                is_updated = True
                print(f"Missing key '{key}' added with default value")
        
        # 변경사항이 있으면 파일 업데이트
        if is_updated:
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
            print(f"Updated settings file with missing keys")
    
    except Exception as e:
        print(f"Error checking settings file: {str(e)}")
        # 파일이 손상된 경우 기본값으로 새로 생성
        default_settings = {
            'openai_key': '',
            'load_dir': os.path.expanduser('~'),
            'last_save_directory': os.path.expanduser('~'),
            'assistant_id': ''
        }
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(default_settings, f, ensure_ascii=False, indent=4)
        print(f"Recreated settings file due to corruption")
        return False
        
    return True

def validate_settings():
    """설정 파일의 값들이 유효한지 확인"""
    try:
        if not os.path.exists(cfg_path):
            print("Settings file does not exist")
            return False
            
        with open(cfg_path, 'r') as f:
            settings = json.load(f)
            api_key = settings.get('openai_key', '').strip()
            
            if not api_key:
                print("API key is empty")
                return False
                
            return True
            
    except Exception as e:
        print(f"Error validating settings: {str(e)}")
        return False

def main():
    app = QApplication(sys.argv)
    
    # 메인 윈도우 생성
    window = MainUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
