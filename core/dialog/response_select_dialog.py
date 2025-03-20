from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QRadioButton, QPushButton, QLabel, QStackedWidget,
                             QWidget, QScrollArea, QButtonGroup)
from PyQt5.QtCore import Qt
import pandas as pd
import os


class ResponseSelectorDialog(QDialog):
    def __init__(self, responses_by_image, parent=None):
        """
        responses_by_image: dict
            {
                'image_path': [response1, response2],
                ...
            }
        """
        super().__init__(parent)
        self.responses_by_image = responses_by_image
        self.current_page = 0
        self.total_pages = len(responses_by_image)
        self.selected_responses = {}
        self.button_groups = {}  # 각 페이지의 버튼 그룹을 저장
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("응답 선택")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        main_layout = QVBoxLayout(self)
        
        # 스택 위젯 설정
        self.stack = QStackedWidget()
        
        # 각 이미지에 대한 페이지 생성
        for idx, (image_path, responses) in enumerate(self.responses_by_image.items()):
            page = self.create_page(image_path, responses, idx)
            self.stack.addWidget(page)
        
        main_layout.addWidget(self.stack)
        
        # 네비게이션 버튼
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("이전")
        self.next_btn = QPushButton("다음")
        self.ok_btn = QPushButton("확인")
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addWidget(self.ok_btn)
        
        main_layout.addLayout(nav_layout)
        
        # 버튼 시그널 연결
        self.prev_btn.clicked.connect(self.show_previous)
        self.next_btn.clicked.connect(self.show_next)
        self.ok_btn.clicked.connect(self.accept)
        
        self.update_nav_buttons()

    def create_page(self, image_path, responses, idx):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # 이미지 경로 표시
        path_label = QLabel(f"이미지 {idx + 1}/{self.total_pages}: {image_path}")
        path_label.setWordWrap(True)
        layout.addWidget(path_label)
        
        # 버튼 그룹 생성
        button_group = QButtonGroup(self)
        self.button_groups[image_path] = button_group
        
        # 응답 선택을 위한 라디오 버튼
        for i, response in enumerate(responses):
            response_widget = QWidget()
            response_layout = QHBoxLayout(response_widget)
            
            radio = QRadioButton()
            button_group.addButton(radio)  # 라디오 버튼을 그룹에 추가
            
            response_label = QLabel(response)
            response_label.setWordWrap(True)
            
            response_layout.addWidget(radio)
            response_layout.addWidget(response_label, stretch=1)
            
            layout.addWidget(response_widget)
            
            # 라디오 버튼 시그널 연결
            radio.toggled.connect(
                lambda checked, path=image_path, resp=response: 
                self.on_response_selected(path, resp) if checked else None
            )
        
        return page

    def update_nav_buttons(self):
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < self.total_pages - 1)
        self.ok_btn.setEnabled(len(self.selected_responses) == self.total_pages)

    def show_previous(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.stack.setCurrentIndex(self.current_page)
            self.update_nav_buttons()

    def show_next(self):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.stack.setCurrentIndex(self.current_page)
            self.update_nav_buttons()

    def on_response_selected(self, image_path, response):
        self.selected_responses[image_path] = response
        self.update_nav_buttons()

    def get_selected_responses(self):
        return self.selected_responses

    def save_selected_responses(self):
        """선택된 응답을 엑셀 파일로 저장"""
        try:
            if not self.selected_responses:
                return False

            # 결과를 DataFrame으로 변환
            results = []
            for image_path, response in self.selected_responses.items():
                parts = response.split('|')
                if len(parts) == 7:
                    results.append({
                        '파일명': parts[0],
                        '파일타입': parts[1],
                        '이미지설명': parts[2],
                        '주제및컨셉': parts[3],
                        '인물정보': parts[4],
                        '키워드': parts[5],
                        '주요색상': parts[6]
                    })

            if results:
                df = pd.DataFrame(results)
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                excel_path = f"selected_results_{timestamp}.xlsx"
                df.to_excel(excel_path, index=False)
                return True

        except Exception as e:
            print(f"Error saving selected responses: {e}")
            return False

    def accept(self):
        """확인 버튼 클릭 시 호출"""
        if len(self.selected_responses) != self.total_pages:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "선택 오류", "모든 이미지에 대해 응답을 선택해주세요.")
            return
        
        # 선택된 응답 저장
        if self.save_selected_responses():
            super().accept()
        else:
            QMessageBox.critical(self, "저장 오류", "선택된 응답을 저장하는 중 오류가 발생했습니다.")