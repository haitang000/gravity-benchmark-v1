# GravityBench

面向 2B–14B 小型 LLM 的本地优先基准测试平台。支持 OpenAI-compatible API、GGUF/llama.cpp 与本地 Hugging Face Transformers，提供中英文指令遵循、数学、Python 编程和性能评测，以及历史结果看板和报告导出。

## 快速启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn backend.main:app --reload
```

访问 `http://127.0.0.1:8000`。前端静态资源在 `frontend/`；开发模式使用 `cd frontend; npm install; npm run dev`。

生产构建：

```powershell
cd frontend; npm install; npm run build
cd ..; .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

代码评测使用 Docker 的无网络、限 CPU/内存容器。首次使用前准备 `python:3.12-slim` 镜像；模型代码不会在宿主机解释器中执行。若部署到 Linux GPU 主机，可把 `llama-cpp-python` 按 CUDA 或 ROCm 选项重新构建，并在模型配置中设置 `n_gpu_layers`；Transformers 使用 PyTorch 的 CUDA/ROCm 检测。

API endpoints：`/api/models`、`/api/models/{id}`（PUT 编辑）、`/api/models/{id}/health`、`/api/runs`（`limit`/`offset` 分页）、`/api/runs/{id}`（`result_limit` 限制返回的结果条数）、`/api/runs/{id}/pause`、`/api/runs/{id}/resume`、`/api/runs/{id}/cancel`、`/api/runs/{id}`（DELETE 删除）、`/api/runs/{id}/export/{json|csv|html}`。OpenAPI 文档在 `/docs`。

本地模型安装可选依赖：`pip install -e ".[local]"`。`llama-cpp-python` 的 CUDA/ROCm wheel 或源码构建请按其官方安装方式完成，GravityBench 会在健康检查时显示编译与设备状态。

WebUI 支持直接填写 API key。服务端调用时使用该 key，模型列表接口只返回掩码，运行报告不包含 key。生产环境也可以改用 `api_key_env` 环境变量引用。
