"""SciAssistant 本地启动脚本（SQLite + 自定义 LLM）"""
import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'deepdiver_v2'))

# 加载 SQLite 适配器（必须在 import app 之前）
sys.path.insert(0, os.path.join(BASE_DIR, 'contrib', 'sqlite_backend'))
import adapter

os.environ.setdefault('FLASK_DEBUG', 'true')

import importlib.util
spec = importlib.util.spec_from_file_location("app", os.path.join(BASE_DIR, "app.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

PORT = int(os.environ.get('PORT', 5050))
print(f"\n  SciAssistant 已启动: http://127.0.0.1:{PORT}")
print(f"  数据库: SQLite ({os.environ.get('SQLITE_DB_PATH', '默认路径')})\n")
m.app.run(host='127.0.0.1', port=PORT, debug=False)
