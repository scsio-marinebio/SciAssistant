# 本地部署指南

在本地环境中运行 SciAssistant，支持两种数据库后端。

## 环境要求

- Python 3.11+
- MySQL 5.7+（生产模式）或 SQLite 3（开发模式）
- 兼容 OpenAI API 的 LLM 服务

## 快速开始（SQLite + DeepSeek）

```bash
# 1. 克隆项目
git clone https://github.com/scsio-marinebio/SciAssistant.git
cd SciAssistant

# 2. 安装依赖
python3 -m venv .venv && source .venv/bin/activate
pip install -r deepdiver_v2/requirements.txt

# 3. 初始化数据库
python contrib/sqlite_backend/init_db.py

# 4. 配置 LLM（复制模板后编辑）
cp deepdiver_v2/config/.env.template deepdiver_v2/config/.env
# 编辑 .env 填入 MODEL_REQUEST_URL 和 MODEL_REQUEST_TOKEN
# 参考 docs/LLM_CONFIG.md

# 5. 用提供的启动脚本运行
python contrib/sqlite_backend/run_local.py
```

浏览器打开 `http://127.0.0.1:5050`，注册账号即可使用。

## MySQL 模式（生产）

```bash
# 1. 安装 MySQL
sudo apt install mysql-server

# 2. 创建数据库
mysql -u root -p -e "CREATE DATABASE sciassist CHARACTER SET utf8mb4;"

# 3. 导入表结构
mysql -u root -p sciassist < chatai.sql

# 4. 修改 app.py 中的 DB_CONFIG
# 编辑 DB_CONFIG 字典填入你的 MySQL 连接信息
```

## 常见问题

### Q: 注册时报 500 错误
检查数据库是否已初始化：`ls data/sciassist.db`
如果使用 SQLite 模式，确保在导入 app 之前先导入了 `adapter.py`

### Q: 聊天提示"获取响应失败"
1. 检查 LLM API 密钥和端点是否正确
2. 查看服务器日志中的错误请求地址——端口或模型名可能不匹配
3. 确认前端 `ai_chat.html` 中的模型名与 `.env` 中一致

### Q: 前端页面 404
确保 `chatAi/` 目录在项目根目录下。如果自定义了路径，需要修改 `app.py` 中的静态文件路由。
