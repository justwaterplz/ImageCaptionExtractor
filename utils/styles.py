import os
import subprocess
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_stylesheet(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            stylesheet = file.read()
        return stylesheet
    except Exception as e:
        print(f"스타일시트 로딩 중 오류 발생: {str(e)}")
        return ""

def add_assets(qrc_path=None, py_path=None):
    if qrc_path is None:
        qrc_path = os.path.join(PROJECT_ROOT, 'res', 'resources.qrc')
    if py_path is None:
        py_path = os.path.join(PROJECT_ROOT, 'res', 'resources_rc.py')

    if os.path.exists(qrc_path):
        subprocess.run(['pyrcc5', qrc_path, '-o', py_path])
        print(f"Compiled {qrc_path} to {py_path}")
    else:
        print(f"Error: {qrc_path} not found")



if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'add_assets':
            add_assets()
        elif sys.argv[1] == 'read_stylesheet':
            if len(sys.argv) > 2:
                stylesheet_path = os.path.join(PROJECT_ROOT, sys.argv[2])
                print(read_stylesheet(stylesheet_path))
            else:
                print("Please provide the path to the stylesheet")
        else:
            print("Unknown command. Available commands: add_assets, read_stylesheet")
    else:
        print("Please specify a command. Available commands: add_assets, read_stylesheet")