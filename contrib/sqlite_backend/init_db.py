"""初始化 SQLite 数据库，创建 SciAssistant 所需表结构"""
import sqlite3, os, sys

def init_database(db_path=None):
    if db_path is None:
        db_path = os.environ.get('SQLITE_DB_PATH',
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sciassist.db'))
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username VARCHAR(50) NOT NULL UNIQUE,
      email VARCHAR(100) NOT NULL UNIQUE,
      password VARCHAR(255) NOT NULL,
      avatar VARCHAR(255),
      createtime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      uptimestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS chat_list (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      session_id VARCHAR(50) NOT NULL,
      title VARCHAR(100) NOT NULL,
      create_time TIMESTAMP,
      update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS conversation_detail (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id VARCHAR(50) NOT NULL,
      from_who VARCHAR(4) NOT NULL,
      round INTEGER DEFAULT 1,
      timestamp TIMESTAMP NOT NULL,
      uuid VARCHAR(100),
      content TEXT NOT NULL,
      think_msg TEXT,
      create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      has_report INTEGER DEFAULT NULL,
      report_title VARCHAR(255) DEFAULT '研究报告'
    );
    CREATE TABLE IF NOT EXISTS user_files (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      file_id VARCHAR(64) NOT NULL UNIQUE,
      user_id INTEGER NOT NULL,
      original_filename VARCHAR(255) NOT NULL,
      stored_filename VARCHAR(255) NOT NULL,
      file_path VARCHAR(512) NOT NULL,
      file_size INTEGER NOT NULL,
      file_type VARCHAR(10) NOT NULL,
      status VARCHAR(20) DEFAULT 'processing',
      upload_time TIMESTAMP NOT NULL,
      process_time TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()
    print(f"SQLite 数据库已初始化: {db_path}")

if __name__ == '__main__':
    init_database()
