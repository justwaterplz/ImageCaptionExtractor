import os
import json
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QLabel, 
                           QWidget, QHBoxLayout, QTableWidgetItem, QCheckBox)
from PyQt5.QtCore import Qt, QFileInfo
from PyQt5.QtGui import QImage, QPixmap

class FileOperations:
    def __init__(self, parent_widget, config_file: str):
        self.parent_widget = parent_widget
        self.config_file = config_file
        # 디버깅을 위한 설정 파일 경로 확인
        print(f"FileOperations 설정 파일 경로: {os.path.abspath(config_file)}")
        
        # 설정 파일 즉시 확인
        self.load_dir = self.load_directory()
        print(f"FileOperations 초기화 - 로드 디렉토리: {self.load_dir}")
        self.all_selected = False  # 전체 선택 상태 추적

    def load_files(self):
        """이미지 파일 로드"""
        try:
            # 디렉토리 다시 로드하여 최신 상태 유지
            self.load_dir = self.load_directory()
            print(f"현재 로드 디렉토리: {self.load_dir}")  # 디버깅용 로그 추가
            
            supported_file_format = ["jpg", "jpeg", "png", "bmp"]
            image_filter = "Images (" + " ".join([f"*.{fmt}" for fmt in supported_file_format]) + ")"
            file_names, _ = QFileDialog.getOpenFileNames(
                self.parent_widget, 
                "파일 열기", 
                self.load_dir, 
                image_filter
            )
            
            if file_names:
                # 각 파일을 개별적으로 추가
                for file_name in file_names:
                    self.add_file_to_table(file_name)
                
                # 마지막 디렉토리 저장
                new_directory = os.path.dirname(file_names[0])
                self.save_directory(new_directory)
                self.load_dir = new_directory
                print(f"새 디렉토리로 업데이트: {new_directory}")  # 디버깅용 로그 추가
                
                # 버튼 상태 업데이트
                self.parent_widget.update_button_states()
                
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {str(e)}")
            QMessageBox.critical(self.parent_widget, '오류', f'파일 로드 중 오류가 발생했습니다: {str(e)}')

    def add_file_to_table(self, file_path):
        """테이블에 파일 추가"""
        try:
            table = self.parent_widget.image_table
            row = table.rowCount()
            table.insertRow(row)
            
            # 컨테이너 위젯 생성
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(5, 5, 5, 5)
            
            # 체크박스 추가
            checkbox = QCheckBox()
            checkbox.setProperty("row", row)
            checkbox.stateChanged.connect(
                lambda state, r=row: self.parent_widget.on_checkbox_changed(r, state)
            )
            layout.addWidget(checkbox)
            
            # 썸네일 추가
            thumbnail = self.create_thumbnail(file_path)
            if thumbnail:
                thumbnail_label = QLabel()
                thumbnail_label.setPixmap(thumbnail)
                thumbnail_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(thumbnail_label)
            
            # 파일명 추가
            file_name = QFileInfo(file_path).fileName()
            name_label = QLabel(file_name)
            name_label.setStyleSheet("padding-left: 10px;")
            layout.addWidget(name_label)
            
            # 우측 여백을 위한 스페이서 추가
            layout.addStretch()
            
            # 데이터 저장
            container.setProperty("file_path", file_path)
            
            # 컨테이너를 테이블에 추가
            table.setCellWidget(row, 0, container)
            
            # 행 높이 조정
            table.setRowHeight(row, 100)
            
        except Exception as e:
            print(f"Error adding file to table: {str(e)}")

    def create_thumbnail(self, image_path, size=80):
        """이미지 썸네일 생성"""
        try:
            image = QImage(image_path)
            if image.isNull():
                return None
                
            # 썸네일 크기에 맞게 이미지 스케일링
            scaled_image = image.scaled(
                size, size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            return QPixmap.fromImage(scaled_image)
            
        except Exception as e:
            print(f"Error creating thumbnail: {str(e)}")
            return None

    def get_selected_files(self):
        """선택된 파일 목록 반환"""
        try:
            selected_files = []
            table = self.parent_widget.image_table
            
            for row in range(table.rowCount()):
                container = table.cellWidget(row, 0)  # 첫 번째 (유일한) 열의 컨테이너 위젯
                if container:
                    # 체크박스는 컨테이너의 레이아웃에서 첫 번째 아이템
                    checkbox = container.layout().itemAt(0).widget()
                    if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                        # 파일 경로는 컨테이너의 속성으로 저장되어 있음
                        file_path = container.property("file_path")
                        if file_path:
                            selected_files.append(file_path)
                            print(f"Selected file: {file_path}")  # 디버깅용
            
            print(f"Total selected files: {len(selected_files)}")  # 디버깅용
            return selected_files
            
        except Exception as e:
            print(f"Error in get_selected_files: {str(e)}")
            return []

    def save_files(self, data, file_path):
        """파일 저장"""
        try:
            # Excel, JSON 등 파일 저장 로직
            pass
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {str(e)}")

    def load_directory(self):
        """마지막 사용 디렉토리 로드"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    load_dir = config.get('load_dir', '')
                    if load_dir and os.path.exists(load_dir):  # 경로가 존재하는지 확인
                        print(f"이미지 로드 디렉토리 불러옴: {load_dir}")
                        return load_dir
            
            print("유효한 로드 디렉토리를 찾을 수 없어 기본값 사용")
        except Exception as e:
            print(f"디렉토리 로드 중 오류: {str(e)}")
            
        # 기본 디렉토리 (사용자 홈)
        default_dir = os.path.expanduser('~')
        print(f"기본 디렉토리 사용: {default_dir}")
        
        # 기본 디렉토리가 존재하지 않는 경우 현재 디렉토리 사용
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()
            print(f"홈 디렉토리가 없어 현재 디렉토리 사용: {default_dir}")
            
        return default_dir

    def save_directory(self, directory):
        """현재 디렉토리 저장"""
        try:
            if not os.path.exists(directory):
                print(f"경고: 존재하지 않는 디렉토리 {directory}")
                return False
                
            config = {}
            if os.path.exists(self.config_file):
                try:
                    with open(self.config_file, 'r') as f:
                        config = json.load(f)
                except json.JSONDecodeError:
                    print(f"JSON 파일 손상, 새로 작성합니다")
                    config = {}
                    
            # 이전 값과 다른 경우에만 저장
            if config.get('load_dir') != directory:
                config['load_dir'] = directory
                
                # 설정 파일 쓰기
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=4)
                
                print(f"이미지 로드 디렉토리 저장: {directory}")
                return True
            
            return False
        except Exception as e:
            print(f"디렉토리 저장 중 오류: {str(e)}")
            return False

    def toggle_select_all(self):
        """전체 선택/해제 토글"""
        try:
            table = self.parent_widget.image_table
            self.all_selected = not self.all_selected
            
            # 모든 행의 체크박스 상태 변경
            for row in range(table.rowCount()):
                container = table.cellWidget(row, 0)
                if container:
                    checkbox = container.layout().itemAt(0).widget()  # 첫 번째 위젯은 체크박스
                    if isinstance(checkbox, QCheckBox):
                        checkbox.setChecked(self.all_selected)
            
            # 버튼 상태 업데이트
            self.parent_widget.update_button_states()
            
        except Exception as e:
            print(f"Error in toggle_select_all: {str(e)}")

    def has_selected_files(self):
        """선택된 파일이 있는지 확인"""
        try:
            table = self.parent_widget.image_table
            for row in range(table.rowCount()):
                container = table.cellWidget(row, 0)
                if container:
                    checkbox = container.layout().itemAt(0).widget()
                    if isinstance(checkbox, QCheckBox) and checkbox.isChecked():
                        return True
            return False
        except Exception as e:
            print(f"Error checking selected files: {str(e)}")
            return False