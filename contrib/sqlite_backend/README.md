# SQLite 后端适配器

让 SciAssistant 脱离 MySQL，使用 SQLite 运行。适合本地开发、测试和轻量部署。

## 使用方式

```bash
# 1. 初始化数据库
python contrib/sqlite_backend/init_db.py

# 2. 在 app.py 或启动脚本最开头导入适配器
```

在你的启动脚本中：

```python
import sys
sys.path.insert(0, 'contrib/sqlite_backend')
import adapter  # 此后 pymysql.connect 自动路由到 SQLite

# 然后正常 import app
from app import app
app.run()
```

## 原理

`adapter.py` 通过 monkey-patch 替换 `pymysql` 模块：

- `pymysql.connect()` → SQLite 连接
- `pymysql.cursors.DictCursor` → SQLite Row 对象（已支持类字典访问）
- MySQL 专有语法（AUTO_INCREMENT、ENGINE、enum 等）→ SQLite 兼容语法
- `NOW()` → `datetime('now')`

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SQLITE_DB_PATH` | `data/sciassist.db` | 数据库文件路径 |

## 限制

- 不支持 MySQL 高级特性（存储过程、触发器等）
- 并发写入性能低于 MySQL
- 生产环境建议仍使用 MySQL
