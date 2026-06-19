# 🏛️ 老爷集团 | Laoye Group

> **没有无价值的产业，只有永恒的老爷。**

老爷集团是一家以 AI 重构产业价值的综合性企业集团。立足实业，拥抱技术，我们相信每个产业都值得被重新定义。

## 关于老爷集团

老爷集团（Laoye Group）始于一个简单的信念：传统产业不是包袱，是未被充分挖掘的金矿。从制造业到体育，从品牌运营到 AI 服务，集团以"产业+科技"的双轮驱动模式，持续为不同领域的资产注入新的生命力。

### 集团业务矩阵

| 板块 | 领域 |
|------|------|
| 🏭 **实业运营** | 全球供应链、制造业、品牌管理 |
| ⚽ **体育产业** | 俱乐部运营、赛事分析与预测 |
| 🤖 **AI 服务** | 智能顾问、预测引擎、自动化决策 |
| 🎨 **品牌资产** | 数字资产、媒体与内容 |

## 网站结构

当前仓库为老爷集团官方网站源码，托管于 `laoyeah.win`。

| 路径 | 内容 | 技术 |
|------|------|------|
| `/` | **Landing Page** — 集团形象展示  | HTML + CSS + JS |
| `/worldcup-2026/` | **2026 世界杯预测** — 5模型集成 + 深度分析 | FastAPI + Poisson + Skill |
| `/show/` | **墨迹擦除展示页** — 交互式品牌展示 | Canvas API |
| `/api/` | **预测 API** — 实时比分预测 | 见 engine/ |

## 预测引擎

`engine/` 目录下集成了完整的 2026 世界杯预测系统，包含 5 个开源模型和一个深度分析框架（Skill Context）。

### 架构概览

```
用户请求 → /api/predictions → predict_server.py
                                  ├── strategy_engine.py (Elo + Poisson)
                                  ├── skill_context.py (7维深度分析)
                                  └── 5 个模型 → 熵加权 → 最终预测
```

### 集成模型

本系统集成了 5 个来自 GitHub 的开源世界杯预测模型，覆盖主流方法论：

| 目录 | 模型名称 | 方法论 | 来源 |
|:-----|:---------|:-------|:-----|
| `worldcup-clean/2026-world-cup-predictor` | AndyDu 预测器 | **Dixon-Coles 泊松模型** — 基于历史大赛数据（163场）的完整泊松回归，预测比分分布精确到每个比分 | [Hicruben/world-cup-2026-prediction-model](https://github.com/Hicruben/world-cup-2026-prediction-model) |
| `worldcup-clean/amir42-predictor` | Amir 预测器 | **XGBoost + Elo 混合** — 特征工程结合 Elo 评级，XGBoost 分类器输出 WDL 概率 | [amirzand2002/world-cup-predictor-2026](https://github.com/amirzand2002/world-cup-predictor-2026) |
| `worldcup-clean/federico1809-predictor` | Federico 预测器 | **RNN + Transfermarkt 特征** — 基于球员身价、历史数据的循环神经网络 | [FedericoPonzi/WorldCup2026](https://github.com/FedericoPonzi/WorldCup2026) |
| `worldcup-clean/model_improve` | Mikobinbin 改进版 | **泊松回归 + 蒙特卡洛模拟** — 多因子特征 + 万次模拟推演 | [mikobinbin/World-Cup-2026-Prediction-Model-improve](https://github.com/mikobinbin/World-Cup-2026-Prediction-Model-improve) |
| `worldcup-clean/2026-world-cup-predictor` | AndyDu 集成预测器 | **多模型集成 + 仿真** — 多种模型投票 + 蒙特卡洛锦标赛推演 | [AndyDu0921/wc26-predict](https://github.com/AndyDu0921/wc26-predict) |

### 预测流程

1. **Elo 评级** — 从 `eloratings.net` 获取 48 支参赛球队最新 Elo 分数
2. **WDL 集成** — 5 个模型各自产出胜/平/负概率，经信息熵加权聚合
3. **比分预测** — AndyDu Dixon-Coles 模型产出 >30 种比分的泊松概率分布
4. **Skill 深度分析** — 7 维度定性分析：战术风格 × 攻防对比 × 实力差 × 比赛局势 × 关键对位 × 风险信号 × 最可能剧本

### 快速启动

```bash
cd engine
pip install -r requirements.txt
python3 predict_server.py
```

启动后 API 服务运行于 `http://localhost:8080`。

## 技术栈

| 层 | 技术 |
|:---|:-----|
| **前端** | 纯 HTML + CSS + JavaScript（零框架依赖） |
| **后端** | FastAPI + Uvicorn |
| **预测引擎** | Poisson / XGBoost / RNN / 集成 5 模型 |
| **深度分析** | Skill Context v2（7维评估框架） |
| **部署** | Nginx + Systemd + Let's Encrypt |
| **域名** | Cloudflare DNS（DNS-only） |
| **数据** | eloratings.net（Elo 评级） |

## 许可证

© 2026 老爷集团 (Laoyeah Group)。保留所有权利。

各开源模型遵循其原始许可证，详见各子目录中的 LICENSE 文件。
