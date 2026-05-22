"""
pymysql → sqlite3 适配器

让 SciAssistant 在无 MySQL 环境下运行。导入此模块后自动生效。

用法:
    import sys
    sys.path.insert(0, 'contrib/sqlite_backend')
    import adapter  # 此后 pymysql 调用自动路由到 SQLite

数据库路径通过环境变量 SQLITE_DB_PATH 设置，默认 ./data/sciassist.db
"""
import sqlite3
import sys, re, os

DB_PATH = os.environ.get('SQLITE_DB_PATH', os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sciassist.db'))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


class SQLiteCursor:
    def __init__(self, sqlite_conn):
        self._c = sqlite_conn.cursor()
        self._conn = sqlite_conn

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, args=None):
        sql = self._xlate(sql)
        try:
            if args: self._c.execute(sql, args)
            else:   self._c.execute(sql)
        except Exception as e:
            raise type(e)(f"{e}\n  SQL: {sql[:200]}")

    def fetchone(self):
        row = self._c.fetchone()
        self._c.connection.commit()
        return row

    def fetchall(self):
        rows = self._c.fetchall()
        self._c.connection.commit()
        return rows

    def close(self): pass

    @property
    def lastrowid(self):
        return self._c.lastrowid

    def __iter__(self):
        return self._c.__iter__()

    def _xlate(self, sql):
        s = sql
        for pattern, repl in [
            (r'(?i)AUTO_INCREMENT\s*=\s*\d+', ''),
            (r'(?i)AUTO_INCREMENT', 'AUTOINCREMENT'),
            (r'(?i)CHARACTER SET \w+', ''),
            (r'(?i)COLLATE \w+', ''),
            (r'(?i)ENGINE=\w+', ''),
            (r'(?i)ROW_FORMAT\s*=\s*\w+', ''),
            (r'mediumtext', 'text'),
            (r'tinyint\(1\)', 'integer'),
            (r'bigint', 'integer'),
            (r"enum\([^)]+\)", 'varchar(20)'),
            (r'(?i)datetime', 'timestamp'),
            (r"(?i)DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP", ''),
            (r"(?i)ON UPDATE CURRENT_TIMESTAMP", ''),
            (r"(?i)INDEX \w+\([^)]+\) USING \w+", ''),
            (r'%s', '?'),
        ]:
            s = re.sub(pattern, repl, s)
        s = s.replace('NOW()', "datetime('now')").replace('now()', "datetime('now')")
        return s


class SQLiteConnection:
    def __init__(self, **kwargs):
        self._conn = sqlite3.connect(DB_PATH)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

    def cursor(self):
        return SQLiteCursor(self._conn)

    def commit(self):   self._conn.commit()
    def rollback(self): self._conn.rollback()
    def close(self):    self._conn.close()


class DictCursor(SQLiteCursor):
    pass


class CursorsModule:
    DictCursor = DictCursor


class FakePymysql:
    connect = staticmethod(lambda *a, **kw: SQLiteConnection(**kw))
    cursors = CursorsModule()
    MySQLError = type('MySQLError', (Exception,), {})
    IntegrityError = type('IntegrityError', (MySQLError,), {})
    OperationalError = type('OperationalError', (MySQLError,), {})

    def __getattr__(self, name):
        return None


sys.modules['pymysql'] = FakePymysql()
sys.modules['pymysql.cursors'] = CursorsModule()
