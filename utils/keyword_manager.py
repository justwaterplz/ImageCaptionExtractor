# utils/keyword_manager.py
import pandas as pd
import os
from typing import List, Dict, Optional
from cfg.cfg import resource_path


class KeywordManager:
    _instance = None
    _keywords_cache: Optional[List[str]] = None
    _keywords_by_category: Optional[Dict[str, List[str]]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KeywordManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._keywords_cache is None:
            self.load_keywords()

    def load_keywords(self) -> None:
        """키워드 데이터를 로드하고 캐시"""
        try:
            # 올바른 경로로 수정
            csv_path = os.path.join(os.getcwd(), "res", "excel_csv", "20250115_keyword.csv")
            print(f"Loading keywords from: {csv_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"File exists check: {os.path.exists(csv_path)}")
            
            if not os.path.exists(csv_path):
                # 상위 디렉토리 내용 확인
                parent_dir = os.path.dirname(os.path.dirname(csv_path))
                print(f"Parent directory: {parent_dir}")
                if os.path.exists(parent_dir):
                    print(f"Directory contents: {os.listdir(parent_dir)}")
                raise FileNotFoundError(f"CSV file not found at {csv_path}")
            
            # 여러 인코딩 시도
            encodings = ['cp949', 'euc-kr', 'utf-8']
            for encoding in encodings:
                try:
                    print(f"Trying encoding: {encoding}")
                    keywords_df = pd.read_csv(csv_path, encoding=encoding)
                    self._keywords_cache = keywords_df.values.flatten().tolist()
                    print(f"Keywords loaded successfully with {encoding}: {len(self._keywords_cache)} keywords")
                    self._create_keyword_categories()
                    return
                except UnicodeDecodeError:
                    print(f"Failed with encoding: {encoding}")
                    continue
                except Exception as e:
                    print(f"Error with encoding {encoding}: {str(e)}")
                    continue
                
            raise Exception("Could not load keywords with any encoding")

        except Exception as e:
            print(f"Error loading keywords: {str(e)}")
            self._keywords_cache = []
            self._keywords_by_category = {}

    def _create_keyword_categories(self) -> None:
        """키워드를 카테고리별로 분류"""
        try:
            # 키워드 카테고리 분류
            self._keywords_by_category = {
                'all': self._keywords_cache,
                'subject': [kw for kw in self._keywords_cache if self._is_subject_keyword(kw)],
                'object': [kw for kw in self._keywords_cache if self._is_object_keyword(kw)],
                'action': [kw for kw in self._keywords_cache if self._is_action_keyword(kw)],
                'mood': [kw for kw in self._keywords_cache if self._is_mood_keyword(kw)]
            }
            print(f"Keywords categorized: {[f'{k}: {len(v)}' for k, v in self._keywords_by_category.items()]}")
        except Exception as e:
            print(f"Error categorizing keywords: {str(e)}")
            self._keywords_by_category = {'all': self._keywords_cache}

    def _is_subject_keyword(self, keyword: str) -> bool:
        """주제 관련 키워드 판별"""
        # 주제 키워드 판별 로직 구현
        return True  # 임시 구현

    def _is_object_keyword(self, keyword: str) -> bool:
        """객체 관련 키워드 판별"""
        return True  # 임시 구현

    def _is_action_keyword(self, keyword: str) -> bool:
        """동작 관련 키워드 판별"""
        return True  # 임시 구현

    def _is_mood_keyword(self, keyword: str) -> bool:
        """분위기 관련 키워드 판별"""
        return True  # 임시 구현

    def get_keywords(self, category: str = 'all', limit: int = 1000) -> List[str]:
        """특정 카테고리의 키워드 반환"""
        if not self._keywords_cache:
            self.load_keywords()

        keywords = self._keywords_by_category.get(category, self._keywords_cache)
        return keywords[:limit] if limit else keywords

    def get_keywords_string(self, category: str = 'all', limit: int = 1000) -> str:
        """키워드를 문자열로 반환"""
        keywords = self.get_keywords(category, limit)
        return ', '.join(keywords)

    @property
    def is_loaded(self) -> bool:
        """키워드 로드 여부 확인"""
        return self._keywords_cache is not None and len(self._keywords_cache) > 0