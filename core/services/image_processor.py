# core/services/image_processor.py

import os
import logging
import time  # datetime.time 대신 time 모듈 직접 import
import traceback
import json
import re

from core.dialog.response_select_dialog import ResponseSelectorDialog
from core.dialog.progress_bar_dialog import ProgressBarDialog
from PyQt5.QtWidgets import QProgressDialog, QMessageBox, QDialog, QFileDialog
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from utils.worker_thread import WorkerThread
import pandas as pd
from openai import OpenAI
from PyQt5.QtWidgets import QApplication
import openpyxl


class ImageProcessor(QObject):
    progress_updated = pyqtSignal(int)
    process_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    status_signal = pyqtSignal(str)

    def __init__(self, main_ui, settings_handler):
        super().__init__()
        self.main_ui = main_ui
        self.settings_handler = settings_handler
        self.progress_dialog = None
        self.worker = None
        self.results = []
        self.processed_files = set()  # 처리된 파일 추적을 위한 set 추가
        self.processing_completed = False  # 처리 완료 상태 추적을 위한 플래그 추가
        self.setup_logger()
        
        # 마지막 저장 위치 가져오기
        try:
            self.last_save_directory = self.settings_handler.get_setting('last_save_directory')
            if not self.last_save_directory:  # 설정이 없는 경우
                self.last_save_directory = os.path.expanduser('~')
        except:
            self.last_save_directory = os.path.expanduser('~')
        
        # API 키를 settings_handler에서 가져옴
        self.api_key = self.settings_handler.get_setting('openai_key')
        self.client = None
        
        # API 키가 있을 때만 클라이언트 초기화
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI client: {e}")

    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def process_images(self, image_paths):
        """이미지 처리 시작"""
        try:
            # 새로운 처리 시작 시 초기화
            self.results = []
            self.processed_files.clear()  # 처리된 파일 목록도 초기화
            self.processing_completed = False  # 처리 완료 상태 초기화
            
            # API 키 확인
            self.api_key = self.settings_handler.get_setting('openai_key')
            if not self.api_key:
                raise ValueError("API 키가 설정되지 않았습니다.")

            # 이전 worker가 있다면 정리
            if self.worker:
                self.worker.stop()
                self.worker.wait()
                self.worker = None

            # 이전 progress_dialog가 있다면 정리
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None

            # 진행 상황 다이얼로그 생성 및 표시
            try:
                self.progress_dialog = ProgressBarDialog(len(image_paths), self.main_ui)
                self.progress_dialog.setWindowModality(Qt.ApplicationModal)
                
                # 초기 로그 추가
                self.progress_dialog.add_log(f"총 {len(image_paths)}개의 이미지 처리를 시작합니다.")
                
                # Worker 스레드 생성 - 수정된 방식으로 초기화
                self.worker = WorkerThread(settings_handler=self.settings_handler, image_processor=self)
                if not self.worker:
                    raise ValueError("Worker thread creation failed")
                
                # 이미지 경로를 큐에 추가
                for image_path in image_paths:
                    self.worker.queue.put(image_path)
                
                # 시그널 연결
                self.worker.progress.connect(self.progress_dialog.update_progress)
                self.worker.current_file.connect(self.progress_dialog.update_current_file)
                self.worker.result_signal.connect(self.handle_result)
                self.worker.error.connect(self.handle_error)
                self.worker.finished.connect(self.process_complete)
                # 상태 시그널 연결 - 로그에 실시간 상태 표시
                self.worker.status_signal.connect(self.progress_dialog.add_log)
                self.worker.increment_progress_signal.connect(self.update_progress_incremental)
                
                # 취소 버튼 연결
                self.progress_dialog.cancel_button.clicked.connect(self.worker.stop)
                
                # 다이얼로그 표시
                self.progress_dialog.show()
                QApplication.processEvents()
                
                # Worker 시작
                self.worker.start()

            except Exception as e:
                self.logger.error(f"Error creating progress dialog: {e}")
                raise

        except Exception as e:
            self.logger.error(f"Error in process_images: {e}")
            self.error_occurred.emit(str(e))
            self.cleanup()

    def parse_response(self, response):
        """API 응답 파싱"""
        try:
            print(f"\n[parse_response] 응답 타입: {type(response)}")
            
            # 응답이 None인 경우 처리
            if response is None:
                print("[parse_response] 응답이 None입니다.")
                return None
            
            # 응답이 딕셔너리인 경우 직접 처리
            if isinstance(response, dict):
                print(f"[parse_response] 딕셔너리 응답 처리 중... 키: {list(response.keys())}")
                
                # 필요한 키가 모두 있는지 확인
                required_keys = ['file_type', 'image_desc', 'theme', 'person_info', 'keywords']
                
                if all(key in response for key in required_keys):
                    # 각 필드의 데이터 정리
                    self.clean_response_fields(response)
                    
                    # 새로운 키워드 형식 (vector store와 AI 생성 키워드 분리)
                    keywords_all = response.get('keywords', '')
                    keywords_vector = response.get('keywords_vector', '')
                    keywords_ai = response.get('keywords_ai', '')
                    
                    print(f"[parse_response] 정리된 응답 데이터: {response}")
                    print(f"[parse_response] 키워드(전체): {keywords_all}")
                    print(f"[parse_response] 키워드(vector store): {keywords_vector}")
                    print(f"[parse_response] 키워드(AI): {keywords_ai}")
                    
                    # 결과 딕셔너리로 반환
                    result = {
                        '파일타입': response.get('file_type', '알 수 없음'),
                        '이미지설명': response.get('image_desc', '설명 없음'),
                        '주제및컨셉': response.get('theme', '정보 없음'),
                        '인물정보': response.get('person_info', '인물 없음'),
                        '키워드목록': keywords_all if keywords_all else '키워드 없음',
                        '키워드(vector store 참조)': keywords_vector,
                        '키워드(AI 생성)': keywords_ai,
                        '주요색상': response.get('colors', '색상 정보 없음')
                    }
                    
                    print("[parse_response] 파싱 결과:")
                    for key, value in result.items():
                        print(f"  - {key}: {value}")
                    
                    return result
                else:
                    missing_keys = [key for key in required_keys if key not in response]
                    print(f"[parse_response] 응답에 필수 키가 없음: {missing_keys}")
                    print(f"[parse_response] 응답 내용: {response}")
                    
                    # 필수 키가 없더라도 기본값으로 결과 생성
                    result = {
                        '파일타입': response.get('file_type', '알 수 없음'),
                        '이미지설명': response.get('image_desc', '설명 없음'),
                        '주제및컨셉': response.get('theme', '정보 없음'),
                        '인물정보': response.get('person_info', '인물 없음'),
                        '키워드목록': response.get('keywords', '키워드 없음'),
                        '키워드(vector store 참조)': response.get('keywords_vector', ''),
                        '키워드(AI 생성)': response.get('keywords_ai', ''),
                        '주요색상': response.get('colors', '색상 정보 없음')
                    }
                    return result
            
            # 응답이 문자열인 경우 처리
            elif isinstance(response, str):
                print(f"[parse_response] 문자열 응답 처리 중... 길이: {len(response)}")
                
                # 응답이 빈 문자열인 경우
                if not response.strip():
                    print("[parse_response] 응답이 빈 문자열입니다.")
                    return {
                        '파일타입': '알 수 없음',
                        '이미지설명': '설명 없음',
                        '주제및컨셉': '정보 없음',
                        '인물정보': '인물 없음',
                        '키워드목록': '키워드 없음',
                        '키워드(vector store 참조)': '',
                        '키워드(AI 생성)': '',
                        '주요색상': '색상 정보 없음'
                    }
                
                # 파이프(|) 문자로 분리된 응답 처리
                if '|' in response:
                    print("[parse_response] 파이프 문자로 구분된 응답 처리")
                    
                    # 응답의 줄바꿈 제거 및 공백 정리
                    clean_response = ' '.join(response.split())
                    
                    # 파이프로 분리하고 앞뒤 공백 제거
                    parts = [part.strip() for part in clean_response.split('|') if part.strip()]
                    print(f"[parse_response] 파이프로 분리된 부분: {len(parts)}개")
                    
                    if len(parts) >= 6:  # 파일명 제외하고 6개 컬럼
                        # 파싱된 데이터 딕셔너리로 변환
                        file_type = parts[0].strip()
                        image_desc = parts[1].strip()
                        theme = parts[2].strip()
                        person_info = parts[3].strip()
                        keywords = parts[4].strip()
                        colors = parts[5].strip()
                        
                        # 키워드 파싱 - 전체키워드[vector store 키워드][AI 생성 키워드] 형식 확인
                        keywords_all = keywords
                        keywords_vector = ""
                        keywords_ai = ""
                        
                        keywords_match = re.search(r'(.*?)\[(.*?)\]\[(.*?)\]', keywords)
                        if keywords_match:
                            keywords_all = keywords_match.group(1).strip()
                            keywords_vector = keywords_match.group(2).strip()
                            keywords_ai = keywords_match.group(3).strip()
                        
                        # 결과 딕셔너리로 반환
                        result = {
                            '파일타입': file_type,
                            '이미지설명': image_desc,
                            '주제및컨셉': theme,
                            '인물정보': person_info,
                            '키워드목록': keywords_all,
                            '키워드(vector store 참조)': keywords_vector,
                            '키워드(AI 생성)': keywords_ai,
                            '주요색상': colors
                        }
                        
                        print("[parse_response] 파이프 구분 파싱 결과:")
                        for key, value in result.items():
                            print(f"  - {key}: {value}")
                        
                        return result
                
                # 파이프 문자가 없는 경우 정규 표현식으로 처리
                print("[parse_response] 파이프 문자가 없는 응답, 정규 표현식으로 처리")
                
                # 원본 줄바꿈 유지하면서 응답 처리
                lines = [line.strip() for line in response.splitlines() if line.strip()]
                print(f"[parse_response] 응답 라인 수: {len(lines)}")
                for i, line in enumerate(lines):
                    print(f"라인 {i+1}: {line}")
                
                # 정규 표현식을 사용하여 파일 유형, 이미지 설명 등 추출
                file_type_match = re.search(r'파일\s*유형:(.+?)(?=\n|$)', response, re.DOTALL | re.IGNORECASE)
                image_desc_match = re.search(r'이미지\s*설명:(.+?)(?=\n\s*주제|$)', response, re.DOTALL | re.IGNORECASE)
                theme_match = re.search(r'주제\s*및\s*컨셉:(.+?)(?=\n\s*인물|$)', response, re.DOTALL | re.IGNORECASE)
                person_match = re.search(r'인물\s*정보:(.+?)(?=\n\s*키워드|$)', response, re.DOTALL | re.IGNORECASE)
                keywords_match = re.search(r'키워드:(.+?)(?=\n\s*주요\s*색상|$)', response, re.DOTALL | re.IGNORECASE)
                colors_match = re.search(r'주요\s*색상:(.+?)(?=\n|$)', response, re.DOTALL | re.IGNORECASE)
                
                # 매치된 결과 출력
                print(f"파일 유형 매치: {file_type_match.group(1).strip() if file_type_match else 'None'}")
                print(f"이미지 설명 매치: {image_desc_match.group(1).strip() if image_desc_match else 'None'}")
                print(f"주제 매치: {theme_match.group(1).strip() if theme_match else 'None'}")
                print(f"인물 정보 매치: {person_match.group(1).strip() if person_match else 'None'}")
                print(f"키워드 매치: {keywords_match.group(1).strip() if keywords_match else 'None'}")
                print(f"색상 매치: {colors_match.group(1).strip() if colors_match else 'None'}")
                
                # 매치된 결과로 딕셔너리 구성
                file_type = file_type_match.group(1).strip() if file_type_match else '알 수 없음'
                image_desc = image_desc_match.group(1).strip() if image_desc_match else '설명 없음'
                theme = theme_match.group(1).strip() if theme_match else '정보 없음'
                person_info = person_match.group(1).strip() if person_match else '인물 없음'
                keywords = keywords_match.group(1).strip() if keywords_match else '키워드 없음'
                colors = colors_match.group(1).strip() if colors_match else '색상 정보 없음'
                
                # 키워드 추가 파싱 - 전체키워드[vector store 키워드][AI 생성 키워드] 형식 확인
                keywords_all = keywords
                keywords_vector = ""
                keywords_ai = ""
                
                keywords_bracket_match = re.search(r'(.*?)\[(.*?)\]\[(.*?)\]', keywords)
                if keywords_bracket_match:
                    keywords_all = keywords_bracket_match.group(1).strip()
                    keywords_vector = keywords_bracket_match.group(2).strip()
                    keywords_ai = keywords_bracket_match.group(3).strip()
                
                # 결과 딕셔너리로 반환
                return {
                    '파일타입': file_type,
                    '이미지설명': image_desc,
                    '주제및컨셉': theme,
                    '인물정보': person_info,
                    '키워드목록': keywords_all,
                    '키워드(vector store 참조)': keywords_vector,
                    '키워드(AI 생성)': keywords_ai,
                    '주요색상': colors
                }
            
            # 지원되지 않는 타입
            else:
                print(f"[parse_response] 지원되지 않는 응답 타입: {type(response)}")
                return {
                    '파일타입': '알 수 없음',
                    '이미지설명': '설명 없음',
                    '주제및컨셉': '정보 없음',
                    '인물정보': '인물 없음',
                    '키워드목록': '키워드 없음',
                    '키워드(vector store 참조)': '',
                    '키워드(AI 생성)': '',
                    '주요색상': '색상 정보 없음'
                }

        except Exception as e:
            print(f"[parse_response] 응답 파싱 오류: {e}")
            traceback.print_exc()
            # 오류 발생 시에도 기본 결과 반환
            return {
                '파일타입': '알 수 없음',
                '이미지설명': '설명 없음',
                '주제및컨셉': '정보 없음',
                '인물정보': '인물 없음',
                '키워드목록': '키워드 없음',
                '키워드(vector store 참조)': '',
                '키워드(AI 생성)': '',
                '주요색상': '색상 정보 없음'
            }

    def clean_response_fields(self, response_dict):
        """응답 딕셔너리의 필드를 정리하는 메서드"""
        if not isinstance(response_dict, dict):
            return
            
        # "주요 색상:" 문자열이 키워드나 다른 필드에 포함되어 있는지 확인 및 정리
        for key in ["file_type", "image_desc", "theme", "person_info", "keywords"]:
            if key in response_dict and isinstance(response_dict[key], str):
                value = response_dict[key]
                
                # "주요 색상:" 패턴 체크 및 추출
                if "주요 색상:" in value or "주요색상:" in value:
                    color_in_field = re.search(r'주요\s*색상:?\s*(.*?)$', value)
                    if color_in_field:
                        # 색상 정보를 올바른 필드로 이동
                        response_dict["colors"] = color_in_field.group(1).strip()
                        # 원래 필드에서 색상 정보 제거
                        response_dict[key] = re.sub(r'주요\s*색상:?\s*.*?$', '', value).strip()
                
                # "키워드:" 패턴 체크 및 추출 (인물 정보 등 다른 필드에 있는 경우)
                if key != "keywords" and ("키워드:" in value or "키워드 :" in value):
                    keyword_in_field = re.search(r'키워드:?\s*(.*?)(?=주요\s*색상|$)', value, re.DOTALL)
                    if keyword_in_field:
                        # 키워드 정보를 올바른 필드로 이동
                        response_dict["keywords"] = keyword_in_field.group(1).strip()
                        # 원래 필드에서 키워드 정보 제거
                        response_dict[key] = re.sub(r'키워드:?\s*.*?(?=주요\s*색상|$)', '', value, flags=re.DOTALL).strip()
        
        # 모든 필드의 값을 정리
        for key, value in response_dict.items():
            if isinstance(value, str):
                # 앞뒤 공백 및 과도한 내부 공백 제거
                clean_value = ' '.join(value.split())
                response_dict[key] = clean_value

    def handle_result(self, file_path, response):
        """API 응답 결과 처리"""
        try:
            file_name = os.path.basename(file_path)
            
            # 응답 검증
            if not response:
                raise ValueError(f"Empty response for {file_name}")
            
            # 응답이 딕셔너리인지 확인
            if not isinstance(response, dict):
                raise ValueError(f"Response is not a dictionary: {type(response)}")
            
            # 필수 필드 확인
            required_fields = ["file_type", "image_desc", "theme", "person_info", "keywords", "colors"]
            for field in required_fields:
                if field not in response:
                    raise ValueError(f"Missing required field '{field}' in response")
            
            # 처리된 파일 목록에 추가
            self.processed_files.add(file_path)
            
            # 결과 저장
            self.results.append({
                'file_path': file_path,
                'response': response
            })
            
            # 로그 추가
            if self.progress_dialog:
                self.progress_dialog.add_log(f"\n처리 완료: {file_name}")
                formatted_response = self.format_response(response)
                self.progress_dialog.add_log(f"응답:\n{formatted_response}")
            
            self.logger.info(f"Successfully processed {file_name}")
            
            # 메인 UI에 결과 전달 (테이블에서 항목 삭제를 위해)
            if self.main_ui and hasattr(self.main_ui, 'handle_result'):
                self.main_ui.handle_result(file_path, response)
            
        except Exception as e:
            self.logger.error(f"Error handling result for {file_path}: {e}")
            self.error_occurred.emit(f"파일 처리 오류: {str(e)}")
            # 오류가 발생해도 프로그램이 계속 실행되도록 예외를 다시 발생시키지 않음

    def format_response(self, response):
        """응답을 보기 좋게 포맷팅"""
        try:
            # 딕셔너리인 경우
            if isinstance(response, dict):
                # 키워드 데이터 가져오기
                keywords_all = response.get('keywords', '정보 없음')
                keywords_vector = response.get('keywords_vector', '')
                keywords_ai = response.get('keywords_ai', '')
                
                formatted_text = (
                    f"파일 유형: {response.get('file_type', '정보 없음')}\n"
                    f"이미지 설명: {response.get('image_desc', '정보 없음')}\n"
                    f"주제 및 컨셉: {response.get('theme', '정보 없음')}\n"
                    f"인물 정보: {response.get('person_info', '정보 없음')}\n"
                    f"키워드: {keywords_all}\n"
                )
                
                # 벡터 스토어 키워드가 있는 경우 추가
                if keywords_vector:
                    formatted_text += f"키워드(vector store): {keywords_vector}\n"
                
                # AI 생성 키워드가 있는 경우 추가
                if keywords_ai:
                    formatted_text += f"키워드(AI 생성): {keywords_ai}\n"
                
                formatted_text += f"주요 색상: {response.get('colors', '정보 없음')}"
                
                return formatted_text
            
            # 문자열인 경우 (파이프로 구분된 응답)
            elif isinstance(response, str) and '|' in response:
                parts = [part.strip() for part in response.split('|') if part.strip()]
                if len(parts) >= 6:
                    return (
                        f"파일 유형: {parts[0]}\n"
                        f"이미지 설명: {parts[1]}\n"
                        f"주제 및 컨셉: {parts[2]}\n"
                        f"인물 정보: {parts[3]}\n"
                        f"키워드: {parts[4]}\n"
                        f"주요 색상: {parts[5]}"
                    )
            
            # 그 외의 경우 원본 반환
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Error formatting response: {e}")
            return str(response)

    def handle_error(self, error_msg):
        """에러 처리"""
        if self.progress_dialog:
            self.progress_dialog.add_log(f"오류: {error_msg}")
        self.error_occurred.emit(error_msg)

    def process_complete(self, results=None):
        """처리 완료"""
        # 이미 처리 완료된 경우 중복 실행 방지
        if self.processing_completed:
            print("이미 처리가 완료되었습니다. 중복 호출 무시.")
            return
        
        # 맨 앞에서 처리 완료 상태 설정 (중복 호출 방지)
        self.processing_completed = True
        
        self.logger.info("process_complete 호출됨")
        
        if self.progress_dialog:
            self.progress_dialog.add_log("\n모든 이미지 처리가 완료되었습니다.")
            self.progress_dialog.update_progress(100)

            if self.results:
                # 최종 결과 로깅
                self.logger.info(f"Processing completed. Total results: {len(self.results)}")
                
                # 자동 저장 설정이 활성화된 경우 결과 저장 시도
                if self.settings_handler.get_setting('auto_save', False):
                    # 결과 저장 시도
                    saved_path = self.save_results_to_excel()
                    if saved_path:
                        self.progress_dialog.add_log(f"\n결과가 자동으로 저장되었습니다: {saved_path}")
                    else:
                        self.progress_dialog.add_log("\n자동 저장에 실패했습니다. 수동으로 저장해주세요.")
                else:
                    # 자동 저장이 비활성화된 경우 안내 메시지 표시 후 저장 다이얼로그 표시
                    self.progress_dialog.add_log("\n처리가 완료되었습니다. 결과를 저장합니다...")
                    # 저장 다이얼로그 표시
                    saved_path = self.save_results_to_excel()
                    if saved_path:
                        self.progress_dialog.add_log(f"\n결과가 저장되었습니다: {saved_path}")
                    else:
                        self.progress_dialog.add_log("\n결과 저장이 취소되었습니다.")

            # 시그널 발생
            self.process_finished.emit()
            
            # 완료 메시지를 다이얼로그에 추가
            self.progress_dialog.add_log("\n처리가 완료되었습니다. 창을 닫으려면 닫기 버튼을 클릭하세요.")
            
            # 취소 버튼 텍스트 변경
            self.progress_dialog.cancel_button.setText("닫기")

    def save_results_to_excel(self):
        """처리 결과 저장"""
        try:
            if not self.results:
                self.status_signal.emit("저장할 결과가 없습니다.")
                return

            # 파일 저장 다이얼로그 표시
            from PyQt5.QtWidgets import QFileDialog
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"키워드추출결과_{timestamp}.xlsx"
            
            excel_file_path, _ = QFileDialog.getSaveFileName(
                self.main_ui,
                "결과 저장",
                os.path.join(self.last_save_directory, default_filename),
                "Excel Files (*.xlsx);;All Files (*)"
            )
            
            if not excel_file_path:  # 사용자가 취소한 경우
                return
                
            # 확장자 확인 및 추가
            if not excel_file_path.lower().endswith('.xlsx'):
                excel_file_path += '.xlsx'
                print(f"Added missing .xlsx extension: {excel_file_path}")
            
            # 선택된 디렉토리 저장
            self.last_save_directory = os.path.dirname(excel_file_path)
            self.settings_handler.save_setting('last_save_directory', self.last_save_directory)

            # 모든 결과를 하나의 DataFrame으로 변환
            all_results = []
            for result in self.results:
                image_path = result['file_path']
                response = result['response']
                
                # 응답이 딕셔너리인지 확인
                if isinstance(response, dict):
                    # 키워드 분리 - 전체키워드[vector store 키워드][AI 생성 키워드] 형식 확인
                    keywords_all = response.get('keywords', '키워드 없음')
                    keywords_vector = response.get('keywords_vector', '')
                    keywords_ai = response.get('keywords_ai', '')
                    
                    df = pd.DataFrame([{
                        '파일명': os.path.basename(image_path),
                        '파일타입': response.get('file_type', '포토'),
                        '이미지설명': response.get('image_desc', '설명 없음'),
                        '주제및컨셉': response.get('theme', '정보 없음'),
                        '인물정보': response.get('person_info', '인물 없음'),
                        '키워드(전체)': keywords_all,
                        '키워드(vector store 참조)': keywords_vector,
                        '키워드(AI 생성)': keywords_ai,
                        '주요색상': response.get('colors', '색상 정보 없음')
                    }])
                    all_results.append(df)

            if all_results:
                # DataFrame 생성 및 저장
                combined_df = pd.concat(all_results, ignore_index=True)
                
                # 저장할 열 순서 지정
                columns = ['파일명', '파일타입', '이미지설명', '주제및컨셉', '인물정보', 
                          '키워드(전체)', '키워드(vector store 참조)', '키워드(AI 생성)', '주요색상']
                
                # 필요한 열만 선택 (누락된 열이 있을 경우 오류 방지)
                available_columns = [col for col in columns if col in combined_df.columns]
                combined_df = combined_df[available_columns]
                
                # 엑셀 저장 시 스타일 적용
                with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
                    # 데이터프레임의 텍스트 컬럼 데이터 정제
                    for col in combined_df.columns:
                        # 긴 텍스트 데이터를 정리하기 위한 처리
                        if col in ['이미지설명', '주제및컨셉', '인물정보', '키워드(전체)', '키워드(vector store 참조)', '키워드(AI 생성)']:
                            combined_df[col] = combined_df[col].apply(lambda x: self.clean_text_for_excel(x) if isinstance(x, str) else x)
                    
                    combined_df.to_excel(writer, index=False, sheet_name='분석결과')
                    
                    # 워크시트 가져오기
                    worksheet = writer.sheets['분석결과']
                    
                    # 열 너비 자동 조정
                    for idx, column in enumerate(worksheet.columns):
                        max_length = 0
                        column_name = combined_df.columns[idx]
                        
                        # 컬럼명의 길이 확인
                        max_length = max(max_length, len(str(column_name)))
                        
                        # 해당 컬럼의 모든 데이터 길이 확인
                        for cell in column:
                            try:
                                if cell.value:
                                    # 줄바꿈이 있는 경우 각 줄의 최대 길이 확인
                                    cell_text = str(cell.value)
                                    lines = cell_text.split('\n')
                                    for line in lines:
                                        # 한글은 2자, 영문/숫자는 1자로 계산
                                        length = sum(2 if ord(c) > 127 else 1 for c in line)
                                        max_length = max(max_length, length)
                            except:
                                pass
                        
                        # 열 너비 설정 (최소 15, 최대 80)
                        if column_name in ['이미지설명', '주제및컨셉']:
                            # 설명/컨셉은 넓게 설정
                            adjusted_width = 80
                        elif column_name in ['인물정보', '키워드(전체)', '키워드(vector store 참조)', '키워드(AI 생성)']:
                            # 키워드와 인물 정보는 중간 크기로 설정
                            adjusted_width = 50
                        else:
                            # 다른 열은 내용에 따라 자동 조정 (최소 15, 최대 30)
                            adjusted_width = max(min(max_length + 2, 30), 15)
                        
                        worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

                    # 행 높이 자동 조정
                    for idx, row in enumerate(worksheet.rows):
                        if idx > 0:  # 헤더 행 제외
                            # 행 높이를 자동으로 조정 (키워드 개행에 따라 적절한 높이 설정)
                            # 키워드 컬럼의 줄 수에 따라 높이 조정
                            row_height = 20  # 기본 높이
                            
                            # 해당 행의 키워드 셀 확인
                            for cell in row:
                                col_idx = cell.column - 1  # 0-based index
                                if col_idx < len(available_columns):
                                    col_name = available_columns[col_idx]
                                    if cell.value and isinstance(cell.value, str):
                                        # 키워드 컬럼은 더 많은 공간 할당
                                        if '키워드' in col_name:
                                            # 키워드 수에 따라 높이 조정 (쉼표 기준)
                                            keyword_count = cell.value.count(',') + 1
                                            cell_height = min(max(keyword_count * 5, 20), 40)  # 최소 20, 최대 40
                                            row_height = max(row_height, cell_height)
                                        else:
                                            # 다른 컬럼은 기본 높이 유지
                                            row_height = max(row_height, 20)
                            
                            worksheet.row_dimensions[idx + 1].height = row_height

                self.logger.info(f"Combined results saved to: {excel_file_path}")
                if self.progress_dialog:
                    self.progress_dialog.add_log(f"\n결과가 저장되었습니다: {excel_file_path}")
                
                return excel_file_path  # 성공적으로 저장된 파일 경로 반환

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            self.error_occurred.emit(f"결과 저장 중 오류 발생: {str(e)}")
            return None  # 오류 발생 시 None 반환

    def cancel_processing(self):
        """처리 취소"""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        self.cleanup()
        self.error_occurred.emit("처리가 취소되었습니다.")

    def cleanup(self):
        """리소스 정리"""
        try:
            if self.worker:
                self.worker.stop()
                self.worker.wait()
                self.worker = None
            
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
            
        except Exception as e:
            self.logger.error(f"Error in cleanup: {e}")

    def set_api_key(self, api_key):
        """API 키 설정"""
        self.api_key = api_key
        # OpenAI 클라이언트 초기화 또는 업데이트
        self.client = OpenAI(api_key=api_key)

    def update_progress_incremental(self):
        """진행 상황을 증가시키는 메소드"""
        if not self.progress_dialog:
            print("Progress dialog not found!")
            return
            
        # 이미 처리 완료된 경우는 중복 호출하지 않음
        if self.processing_completed:
            return
            
        # 현재 처리된 항목 수 계산
        processed_count = len(self.processed_files)
        total_count = self.progress_dialog.total_images
        
        # 디버깅을 위한 로그 추가
        print(f"DEBUG - Processed files: {processed_count}/{total_count}")
        print(f"DEBUG - Processed files list: {self.processed_files}")
        
        # 진행률 계산 (0-100%)
        if total_count > 0:
            progress_percentage = min(100, int((processed_count / total_count) * 100))
            print(f"DEBUG - Progress percentage: {progress_percentage}%")
            self.progress_updated.emit(progress_percentage)
            self.progress_dialog.update_progress(progress_percentage)
            
            # 처리 로그 업데이트
            self.progress_dialog.add_log(f"진행 상황: {processed_count}/{total_count} 파일 처리 완료 ({progress_percentage}%)")
        
        # 모든 항목 처리 완료 확인 (이미 완료 상태가 아닌 경우만)
        if not self.processing_completed and processed_count >= total_count:
            print("DEBUG - All files processed, calling process_complete")
            self.process_complete()

    def clean_text_for_excel(self, text):
        """엑셀에 표시할 텍스트 정리"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # 줄바꿈 문자를 엑셀에서 사용할 수 있는 줄바꿈으로 변환
        # \n을 줄바꿈으로 변환 (엑셀에서는 Alt+Enter로 표시됨)
        text = re.sub(r'\\n', '\n', text)
        
        # 키워드 컬럼인 경우 특별 처리
        if '키워드' in text and ',' in text:
            # 키워드 형식을 유지하되 가독성을 위해 개행 추가
            # 쉼표 뒤에 공백이 있는 경우와 없는 경우 모두 처리
            text = re.sub(r',\s*', ', ', text)  # 쉼표 뒤에 공백 추가
            
            # 키워드 형식이 "키워드 (가중치)" 형태인지 확인하고 유지
            text = re.sub(r'\(\s*([0-9.]+)\s*\)', r' (\1)', text)  # 괄호 안의 공백 제거
        
        # 과도한 공백 제거 (선행/후행 공백 및 여러 공백을 하나로 병합)
        # 줄바꿈은 유지하면서 각 줄의 앞뒤 공백만 제거
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines]
        text = '\n'.join(cleaned_lines)
        
        # 특수 문자 중 엑셀에서 문제가 될 수 있는 것들 처리
        # 유니코드 제어 문자 제거
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        # 매우 긴 텍스트의 경우, 첫 1000자만 유지 (필요시 조정)
        if len(text) > 1000:
            text = text[:997] + '...'
        
        return text