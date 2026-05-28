
##  项目简介

SciAssistant 是一个基于大语言模型的智能研究助手系统，通过多智能体协作架构，帮助用户进行深度信息检索、文档分析和研究报告生成。

###  核心特点

- **多智能体协作** - Planner、Information Seeker、Writer 三大智能体协同工作
- **智能文档处理** - 本地文件库创建，支持 PDF、Word、txt等多种格式
- **深度信息检索** - 批量网络搜索、网页爬取
- **专业报告生成** - 自动生成结构化研究报告，支持 Markdown 和 PDF 导出
- **会话管理** - 多会话支持，历史记录追溯
- **模式概览** - Chat (普通对话)/Reasoner(深度推理)/DeepDiver(万字长文)模式
- **MCP server 开放空间搜索服务** - 本地知识库、Google search、PubMed API、ArXiv API等


### Powered by openPangu

本项目基于盘古DeepDiver-V2深度增量开发，采用多智能体协作架构（Multi-Agent System）
，提供完整的前后端服务化解决方案。

核心特性：
* 智能体协同编排 - Planner、Information Seeker、Writer 三大智能体

* 服务化封装 - RESTful API + WebSocket 实时通信

* 前后端一体 - Flask/FastAPI 后端 + 现代化 Web 前端

* 支持用户管理、会话管理、文档处理

  盘古DeepDiver-V2参考链接：https://ai.gitcode.com/ascend-tribe/openPangu-Embedded-7B-DeepDiver

---

##  功能特性

###  多智能体系统

```
┌─────────────────┐
│ Planner Agent   │  任务规划与分解
└────────┬────────┘
         │
    ┌────▼────┐
    │ Tasks   │
    └────┬────┘
         │
    ┌────▼──────────────────┐
    │                       │
┌───▼──────────────┐  ┌────▼─────────┐
│ Info Seeker      │  │ Writer       │
│ - 网络搜索       │  │ - 报告撰写   │
│ - 网页爬取       │  │ - 内容生成   │
│ - 文档提取       │  │ - 格式输出   │
└──────────────────┘  └──────────────┘
```

### 模式概览
*   **Chat (普通对话)**: 标准的 LLM 对话模式，直接与大模型交互。
*   **Reasoner (深度推理)**: 针对支持 "思维链 (Chain of Thought)" 的模型设计（如 DeepSeek-R1, Pangu-Reasoner），界面会展示模型的思考/推理过程 (`reasoning_content`)。
*   **DeepDiver (万字长文)**: 调用后端 Multi-Agent 系统，执行复杂的长文写作和深度信息检索任务。

###  文档处理能力

| 功能 | 说明 |
|------|------|
| **多格式支持** | PDF, Word,  TXT|
| **文档库** | 用户专属文档空间，支持批量管理 |

### 信息检索

- **批量网络搜索** - 多关键词并行搜索，智能去重排序
- **网页爬取** - 异步并发爬取，智能内容提取
- **搜索结果分类** - 自动分类搜索结果，提取关键信息

###  报告生成

- **结构化输出** - 自动生成目录、章节、引用
- **多格式导出** - Markdown（便于编辑）、PDF（专业排版）
- Title标题/abstract摘要/KeyWords关键词/Citation参考文献生成
- PDF生成优先使用用户系统自带字体以提升加载性能，同时内置了开源字体作为备用方案

### 会话管理

- **多会话支持** - 每个研究任务独立会话
- **历史记录** - 完整的对话历史和操作记录
- **文件关联** - 会话级别的文件上传和管理

### 用户系统

- **用户认证** - 注册、登录、密码重置
- **JWT 令牌** - 无状态身份验证，支持"记住我"
- **权限控制** - 会话隔离、文档访问控制
- **安全加密** - SHA-256 密码加密，SQL 注入防护

---

## 快速开始

### 环境要求

- Python 3.8+
- MySQL 5.7+ / 8.0+
- 操作系统：Windows / Linux 

### 安装

```bash
# 1. 克隆项目
git clone <repository-url>
cd SciAssistant

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp config/env.template config/.env
# 编辑 config/.env 文件，配置数据库和 API 密钥

# 4. 初始化数据库
mysql -u root -p < database/schema.sql

# 5. 启动 MCP 服务器
python src/tools/mcp_server_standard.py --config src/tools/server_config.yaml

# 6. 启动 Flask Web API（用户管理、会话管理）
python app.py

# 7. 启动 PlannerAgent HTTP 服务器（智能体任务处理）
python cli/a.py
```



---

## 系统架构

### 技术栈

| 类别 | 技术 |
|------|------|
| **Web 框架** | Flask (用户管理), FastAPI (智能体服务), Flask-CORS |
| **数据库** | MySQL, PyMySQL |
| **AI/LLM** | OpenPangu |
| **HTTP 客户端** | httpx, aiohttp, requests |
| **文档处理** | pdfminer.six, PyPDF2, ReportLab |
| **认证** | PyJWT |
| **其他** | python-dotenv, Rich, Pydantic |

### 项目结构

```
SciAssistant

chatAi/
├── ai_chat.html               # 主页
├── forgot-password.html       # 忘记密码
├── login.html                 # 登录
├── register.html              # 注册

deepdiver_v2/
├── app.py                    # Flask Web API 入口（用户管理、会话管理）
├── requirements.txt          # 项目依赖
├── LICENSE.txt              # Apache-2.0 许可证
├── NOTICE                   # 版权声明
│
├── cli/                     # 命令行工具和服务
│   ├── demo.py          	 # 命令行入口
│   ├── run_demo.sh         
│   └── a.py                # PlannerAgent HTTP 服务器（FastAPI）
│
├── config/                  # 配置文件
│   ├── config.py           # 配置管理
│   ├── logging_config.py   # 日志配置
│   └── .env                # 环境变量
│
├── src/
│   ├── agents/             # 智能体
│   │   ├── base_agent.py
│   │   ├── planner_agent.py
│   │   ├── objective_information_seeker.py
│   │   ├── subjective_information_seeker.py
│   │   └── writer_agent.py
│   │
│   ├── tools/              # 工具集
│   │   ├── mcp_tools.py
│   │   ├── mcp_client.py
│   │   ├── mcp_server_standard.py
│   │   └── api_tool.py
│   │
│   └── utils/              # 工具函数
│       ├── status_codes.py
│       └── task_manager.py  # 任务管理器
│
├── workspaces/             # 工作空间（持久化）
├── uploads/                # 临时上传
├── user_files/            # 用户文档库
└── logs/                  # 日志文件
```

---


## 配置说明

编辑 `config/.env` 文件，参考典型配置如下：

```bash
# ================= LLM 模型配置 =================
# 你的大模型 API 地址
MODEL_REQUEST_URL=http://your-llm-endpoint/v1/chat/completions
# 模型 API Token
MODEL_REQUEST_TOKEN=your-service-token
# 模型名称
MODEL_NAME=your-model-name

# ================= MCP 服务器配置 =================
# MCP Server 地址 (默认为本机 6274 端口)
MCP_SERVER_URL=http://localhost:6274/mcp
MCP_USE_STDIO=false

# ================= 搜索与爬虫配置 =================
# 搜索引擎 API (如 Bing/Google Custom Search)
SEARCH_ENGINE_BASE_URL=https://google.serper.dev/search
SEARCH_ENGINE_API_KEYS=your-google-key

# URL 爬虫配置 (用于读取网页内容)
URL_CRAWLER_BASE_URL=http://your-crawler-api
URL_CRAWLER_API_KEYS=your-api-key
URL_CRAWLER_MAX_TOKENS=100000

# ================= Agent 迭代限制 =================
PLANNER_MAX_ITERATION=40
INFORMATION_SEEKER_MAX_ITERATION=30
WRITER_MAX_ITERATION=40

# ================= 模式 ================= 
PLANNER_MODE=writing # auto, writing, qa

# ================= 路径配置 =================
# 工作区路径 (绝对路径或相对路径)
TRAJECTORY_STORAGE_PATH=./workspace
REPORT_OUTPUT_PATH=./report

# ================= 常规设置 =================
DEBUG_MODE=false
MAX_RETRIES=3
TIMEOUT=30

```



编辑`/app.py`文件，参考典型配置如下：
```bash
# 数据库配置
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your-password
MYSQL_DATABASE=chatai
```



编辑`src/tools/server_config.yaml`文件，参考典型配置如下：

```yaml
# MCP 服务器配置
server:
  host: 127.0.0.1
  port: 6274
  session_ttl_seconds: 3600
  max_sessions: 1000
  rate_limit_requests_per_minute: 300

tool_rate_limits:
  batch_web_search:
    requests_per_minute: 60
    requests_per_hour: 500
  url_crawler:
    requests_per_minute: 30
    requests_per_hour: 300
```

`\src\tools\mcp_tools.py`文件，找到相关字体放到该目录下：

```python
mcp_tools字体路径：
# Linux 字体路径（可能需要安装中文字体包）
	simsun_path = "/usr/share/fonts/dejavu/SIMSUN.TTC"
	simhei_path = "/usr/share/fonts/dejavu/SIMHEI.TTF"
	arial_path = "/usr/share/fonts/dejavu/ARIAL.TTF"  # Linux Arial path (if installed)
	symbol_path = "/usr/share/fonts/dejavu/DejaVuSans.ttf"  # Linux fallback
# 使用黑白emoji字体（多个可能的路径）
	emoji_paths = [
		"/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",  # Noto Emoji 黑白版本
		"/usr/share/fonts/noto/NotoEmoji-Regular.ttf",
		"/usr/share/fonts/google-noto-emoji/NotoEmoji-Regular.ttf",
		symbol_path  # 最后的备选方案：DejaVuSans
	]
```

---



## API 接口

### 用户认证

```bash
# 注册
POST /api/register
{
  "username": "user@example.com",
  "email": "user@example.com",
  "password": "password123",
  "confirmPassword": "password123"
}

# 登录
POST /api/login
{
  "loginId": "user@example.com",
  "password": "password123",
  "remember_me": false
}
```

### 会话管理

```bash
# 创建会话
POST /api/chat/sessions
{
  "user_id": "user-id",
  "title": "研究主题"
}

# 获取会话列表
GET /api/chat/sessions/<user_id>

# 更新会话标题
PUT /api/chat/sessions/<session_id>
{
  "title": "新标题"
}

# 删除会话
DELETE /api/chat/sessions/<session_id>
```

### 消息交互

```bash
# 发送消息
POST /api/chat/messages
{
  "session_id": "session-id",
  "from_who": "user",
  "content": "请帮我研究人工智能在医疗领域的应用",
  "round": 1
}

# 获取会话消息
GET /api/chat/messages/<session_id>
```

### 文档管理

```bash
# 上传文档（临时）
POST /api/context/upload
Content-Type: multipart/form-data
file: <file-data>
user_id: <user-id>
save_to_library: true

# 上传到文档库
POST /api/files/upload
Content-Type: multipart/form-data
files: <file-data>
user_id: <user-id>

# 获取文档列表
GET /api/files/list/<user_id>

# 删除文档
DELETE /api/files/delete/<file_id>

# 批量删除
POST /api/files/batch-delete
{
  "file_ids": ["file-id-1", "file-id-2"]
}
```


### 报告下载

```bash
# 下载 PDF 报告
GET /api/download_pdf?session_id=<session-id>

# 下载 Markdown 报告
GET /api/download_md?session_id=<session-id>
```

### PlannerAgent 服务（智能体任务处理）

```bash
# 处理单个查询（智能体任务）
POST http://localhost:8000/api/query
{
  "query": "请帮我研究人工智能在医疗领域的应用",
  "taskId": "task-123",
  "user_files": [
    {"file_id": "file-1", "filename": "document.pdf"}
  ],
  "reference_files": [
    {"file_id": "file-2", "filename": "reference.pdf"}
  ],
  "use_web_search": true,
  "prioritize_user_files": true,
  "username": "用户名"
}

# 获取任务状态
GET http://localhost:8000/api/task/{task_id}

# 取消正在运行的任务
POST http://localhost:8000/api/task/{task_id}/cancel

# 获取并发状态
GET http://localhost:8000/api/concurrency

# 获取服务器状态
GET http://localhost:8000/api/status
```


