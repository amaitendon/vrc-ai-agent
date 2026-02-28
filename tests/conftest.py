import os
import sys
import pytest
import asyncio

# プロジェクトルートにパスを通す
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.context import AppContext


@pytest.fixture
def clean_app_context():
    """AppContextのシングルトンをリセットするフィクスチャ"""
    AppContext._instance = None
    ctx = AppContext.get()
    return ctx


@pytest.fixture
def mock_priority_queue():
    """テスト用のPriorityQueue"""
    return asyncio.PriorityQueue()
