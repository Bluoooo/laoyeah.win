# 🏛️ 老爷集团 | Laoye Group

以 AI 重构产业价值，让技术归于实业。

## 站点结构

| 路径 | 内容 |
|------|------|
| `/` | Landing Page（倒计时 + 世界地图） |
| `/worldcup-2026/` | 世界杯预测（5模型集成） |
| `/show/` | 墨迹擦除展示页 |
| `/666/` | 梅西彩蛋页 |

## 预测后端

```bash
cd engine
pip install -r requirements.txt
python3 predict_server.py
```

启动后访问 `http://localhost:8080/api/predictions`。

## 技术栈

- 纯前端：HTML + CSS + JavaScript（无框架依赖）
- 后端：FastAPI + Uvicorn
- 预测引擎：5 模型集成（Dixon-Coles / XGBoost / RNN / Poisson）
- 部署：Nginx + Systemd
