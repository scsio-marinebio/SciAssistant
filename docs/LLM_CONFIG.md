# 多 LLM 后端配置指南

SciAssistant 通过 litellm 支持接入任意 OpenAI 兼容 API。本文档说明如何配置不同的 LLM 后端。

## 配置方式

编辑 `deepdiver_v2/config/.env`：

```bash
MODEL_REQUEST_URL=<API 端点>
MODEL_REQUEST_TOKEN=<API 密钥>
MODEL_NAME=<模型名称>
```

## DeepSeek

```bash
MODEL_REQUEST_URL=https://api.deepseek.com/v1/chat/completions
MODEL_REQUEST_TOKEN=sk-your-key
MODEL_NAME=deepseek-v4-pro
```

## 通义千问 (Qwen)

通过阿里云百炼平台：

```bash
MODEL_REQUEST_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
MODEL_REQUEST_TOKEN=sk-your-key
MODEL_NAME=qwen-plus
```

## OpenAI

```bash
MODEL_REQUEST_URL=https://api.openai.com/v1/chat/completions
MODEL_REQUEST_TOKEN=sk-your-key
MODEL_NAME=gpt-4o
```

## 本地模型 (Ollama)

```bash
MODEL_REQUEST_URL=http://localhost:11434/v1/chat/completions
MODEL_REQUEST_TOKEN=ollama
MODEL_NAME=qwen2.5:14b
```

## 前端模型名修改

如果使用非盘古模型，需同步修改前端 JavaScript 中的模型名。在 `chatAi/ai_chat.html` 中搜索 `"model"`：

```javascript
// 将所有 "model":"pangu" 或 "model":"pangu_auto" 改为你的模型名
"model":"deepseek-v4-pro"
```

## 验证

配置完成后启动服务，在 Chat 模式下发送消息，查看服务器日志确认请求发送到了正确的 API 端点。
