# LEGACY CODE - 더 이상 사용되지 않는 코드입니다.
# core/services/image_processor.py가 대체 파일로 사용됩니다.
# 안전을 위해 주석 처리되었습니다.

# from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
# from PyQt5.QtCore import QObject, pyqtSignal, QThread

# import os
# import json
# import traceback
# from utils.worker_thread import WorkerThread
# from ui.progress_dialog import ProgressDialog

# class ImageProcessor(QObject):
#     progress_signal = pyqtSignal(int, int)  # 처리된 이미지 수, 전체 이미지 수
#     status_signal = pyqtSignal(str)  # 상태 메시지
#     completed_signal = pyqtSignal(str)  # 완료 메시지

#     def __init__(self, api_key, assistant_id):
#         super().__init__()
#         self.api_key = api_key
#         self.assistant_id = assistant_id
#         self.worker = None
#         self.progress_dialog = None

#     def process_images(self, image_paths):
#         """이미지 처리 시작"""
#         try:
#             # 이미지 경로 검증
#             if not image_paths:
#                 print("처리할 이미지가 없습니다.")
#                 return
                
#             if isinstance(image_paths, str):
#                 image_paths = [image_paths]  # 단일 경로를 리스트로 변환
            
#             # 이전 worker가 있으면 정리
#             if hasattr(self, 'worker_thread') and self.worker_thread:
#                 self.worker_thread.exit()
#                 self.worker_thread = None
            
#             # 이전 progress_dialog가 있으면 정리
#             if hasattr(self, 'progress_dialog') and self.progress_dialog:
#                 self.progress_dialog.close()
#                 self.progress_dialog = None
                
#             # 이미지 큐 생성
#             from queue import Queue
#             image_queue = Queue()
            
#             # 큐에 이미지 추가
#             for path in image_paths:
#                 image_queue.put(path)
                
#             # 진행 대화상자 생성
#             from ui.progress_dialog import ProgressDialog
#             self.progress_dialog = ProgressDialog(None)  # 부모 없이 생성
#             self.progress_dialog.setWindowTitle(f"이미지 처리 중 ({len(image_paths)}개)")
#             self.progress_dialog.progress_bar.setMaximum(len(image_paths))
            
#             # 워커 스레드 생성
#             from utils.worker_thread import WorkerThread
#             self.worker_thread = WorkerThread(
#                 queue=image_queue, 
#                 settings_handler=self,
#                 api_key=self.api_key
#             )
            
#             # 워커 스레드에 API 키 설정
#             self.worker_thread.set_credentials(self.api_key, self.assistant_id)
            
#             # 시그널 연결
#             self.worker_thread.progress.connect(self.progress_dialog.update_progress)
#             self.worker_thread.current_file.connect(self.progress_dialog.update_current_file)
#             self.worker_thread.status_signal.connect(self.progress_dialog.add_log)
#             self.worker_thread.result_signal.connect(self.handle_result)
#             self.worker_thread.completed_signal.connect(self.progress_dialog.add_log)
#             self.worker_thread.completed_signal.connect(self.on_completed)
#             self.worker_thread.error_signal.connect(self.handle_error)
            
#             # 버튼 연결
#             self.progress_dialog.cancel_button.clicked.connect(self.worker_thread.cancel)
#             self.progress_dialog.pause_button.clicked.connect(self.worker_thread.pause)
            
#             # 완료 시 처리
#             self.worker_thread.finished.connect(self.on_worker_finished)
            
#             # 다이얼로그 표시 및 스레드 시작
#             self.progress_dialog.show()
#             self.worker_thread.start()
            
#         except Exception as e:
#             print(f"이미지 처리 중 오류 발생: {e}")
#             import traceback
#             traceback.print_exc()

#     def update_progress(self, processed, total):
#         """진행 상황 업데이트"""
#         self.progress_signal.emit(processed, total)

#     def update_status(self, message):
#         """상태 메시지 업데이트"""
#         self.status_signal.emit(message)

#     def on_completed(self, message):
#         """처리 완료 시 호출"""
#         self.completed_signal.emit(message)
        
#         # 완료 메시지가 '취소됨'이 아닌 경우에만 완료 다이얼로그 표시
#         if "취소" not in message.lower():
#             if self.progress_dialog:
#                 self.progress_dialog.label.setText(f"처리 완료: {message}")
#                 # 버튼 텍스트 변경
#                 self.progress_dialog.cancel_button.setText("닫기")
#                 # 일시 정지 버튼 비활성화
#                 self.progress_dialog.pause_button.setEnabled(False)

#     def check_openai_key(self):
#         """OpenAI API 키가 설정되어 있는지 확인"""
#         openai_key = self.settings_handler.get_setting('openai_key', '')
#         return bool(openai_key)

#     def check_assistant_id(self):
#         """Assistant ID가 설정되어 있는지 확인"""
#         assistant_id = self.settings_handler.get_setting('assistant_id', '')
#         return bool(assistant_id)

#     def show_error(self, title, message):
#         """오류 메시지 표시"""
#         QMessageBox.critical(self.parent, title, message)

#     def show_message(self, title, message):
#         """정보 메시지 표시"""
#         QMessageBox.information(self.parent, title, message)

#     def create_progress_dialog(self, total_images):
#         """진행 상황 다이얼로그 생성"""
#         dialog = ProgressDialog(self.parent)
#         dialog.setWindowTitle(f"이미지 처리 중 ({total_images}개)")
#         dialog.progress_bar.setMaximum(total_images)
#         dialog.progress_bar.setValue(0)
#         dialog.add_log(f"처리 준비 중... 총 {total_images}개 이미지")
        
#         return dialog

#     def handle_error(self, error_type, error_message):
#         """에러 처리"""
#         error_msg = f"{error_type}: {error_message}"
#         print(f"에러 발생: {error_msg}")
#         if self.progress_dialog:
#             self.progress_dialog.add_log(f"오류: {error_msg}")
#         else:
#             self.show_error("에러", error_msg)

#     def on_worker_finished(self):
#         """워커 스레드 작업 완료 시 처리"""
#         print("이미지 처리 완료")
#         if self.progress_dialog:
#             self.progress_dialog.enable_close_button()
            
#     def cleanup(self):
#         """리소스 정리"""
#         try:
#             if hasattr(self, 'worker_thread') and self.worker_thread:
#                 # 안전하게 스레드 종료
#                 self.worker_thread.exit()
#                 self.worker_thread = None
                
#             if hasattr(self, 'progress_dialog') and self.progress_dialog:
#                 self.progress_dialog = None
#         except Exception as e:
#             print(f"리소스 정리 중 오류: {e}")
            
#     def __del__(self):
#         """소멸자"""
#         self.cleanup()

#     def settings_handler(self, image_path):
#         # Implementation of settings_handler method
#         pass 

#     def get_setting(self, key, default=None):
#         """설정 값 가져오기 - SettingsHandler 역할 수행"""
#         if key == 'openai_key':
#             return self.api_key
#         elif key == 'assistant_id':
#             return self.assistant_id
#         return default
        
#     def save_setting(self, key, value):
#         """설정 값 저장하기 - SettingsHandler 역할 수행"""
#         # 실제로는 저장하지 않고 클래스 변수에만 임시 저장
#         if key == 'openai_key':
#             self.api_key = value
#         elif key == 'assistant_id':
#             self.assistant_id = value
#         return True
        
#     def handle_result(self, image_path, result):
#         """이미지 처리 결과 처리"""
#         try:
#             print(f"이미지 처리 결과: {image_path}")
#             # 여기서 결과 처리 (UI 업데이트 등)
#         except Exception as e:
#             print(f"결과 처리 중 오류: {e}")
        