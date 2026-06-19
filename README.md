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

## 预测模型

`engine/worldcup-clean/` 下集成了 5 个开源世界杯预测模型：

| 目录 | 来源 | 方法 |
|------|------|------|
| `world-cup-2026-prediction-model` | [Hicruben/world-cup-2026-prediction-model](https://github.com/Hicruben/world-cup-2026-prediction-model) | Dixon-Coles 泊松模型 |
| `amir42-predictor` | [amirzand2002/world-cup-predictor-2026](https://github.com/amirzand2002/world-cup-predictor-2026) | XGBoost + Elo 混合 |
| `federico1809-predictor` | [FedericoPonzi/WorldCup2026](https://github.com/FedericoPonzi/WorldCup2026) | RNN + Transfermarkt 特征 |
| `model_improve` | [mikobinbin/World-Cup-2026-Prediction-Model-improve](https://github.com/mikobinbin/World-Cup-2026-Prediction-Model-improve) | 泊松回归 + 蒙特卡洛 |
| `2026-world-cup-predictor` | [AndyDu0921/wc26-predict](https://github.com/AndyDu0921/wc26-predict) | 多模型集成 + 模拟 |

预测流程：5 模型分别产出 WDL 概率 → 信息熵加权聚合方向 → AndyDu Dixon-Coles 产出比分分布 → Skill 环境参数调制。

## 技术栈

- 纯前端：HTML + CSS + JavaScript（无框架依赖）
- 后端：FastAPI + Uvicorn
- 预测引擎：5 模型集成（Dixon-Coles / XGBoost / RNN / Poisson / Multi-model）
- 部署：Nginx + Systemd
- Elo 数据源：eloratings.net
