# 部署手册

## 1. 本地运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts\train.py --epochs 30
uvicorn gnn_rl_recommender.app:app --host 0.0.0.0 --port 8000
```

访问 `http://127.0.0.1:8000/docs` 调试接口。

## 2. Docker 部署

```bash
docker compose up --build -d
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/v1/recommendations/3?limit=5"
```

## 3. 云服务器部署

服务器安装 Git 与 Docker 后执行：

```bash
git clone https://github.com/fangyunok/gnn-rl-recommender-system.git
cd gnn-rl-recommender-system
docker compose up --build -d
```

生产环境建议在服务前增加 Nginx/Caddy、HTTPS、访问日志和鉴权。模型文件不提交 Git，
可在 CI/CD 中训练或从对象存储下载后挂载到 `/app/artifacts`。

## 4. CI/CD

GitHub Actions 在每次 push 与 pull request 时自动运行静态检查、单元测试和 Docker 镜像构建，
确保仓库始终保持可部署状态。

