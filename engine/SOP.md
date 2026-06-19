# 世界杯预测 SOP - 量化模型 + 深度分析框架

> 标准作业程序：结合 5 模型量化预测系统 + Football Match Deep Analysis Skill 定性框架
> 版本: 1.0 | 最后更新: 2026-06-19

---

## 一、系统架构

```
┌─────────────────────────────────────────────────┐
│                   用户前端                        │
│           laoyeah.win/worldcup-2026/             │
└──────────────────────┬──────────────────────────┘
                       │ HTTPS /api/predictions
┌──────────────────────▼──────────────────────────┐
│          FastAPI 预测服务器 (port 8080)           │
│  - 提供 RESTful API                             │
│  - 每天 8AM cron 自动刷新                        │
│  - 提供 72 场小组赛预测数据                       │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│           v2 量化模型层 (optimized_engine.py)     │
├──────────────────────────────────────────────────┤
│ Hicruben_v2    DC-RHO=-0.25   Dixon-Coles        │
│ mikobinbin_v2  DC-RHO=-0.20   Poisson            │
│ AndyDu_v2      draw=0.40cap  混合 Poisson/logistic│
│ amir42         XGBoost       ML 分类             │
│ federico1809   XGBoost+聚类  唯一平局模型          │
├──────────────────────────────────────────────────┤
│       后处理校准 + 自适应集成权重                  │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│       深度分析层 (Deep Analysis Framework)        │
├──────────────────────────────────────────────────┤
│ - 球队风格分类（超级巨星型/体系型/黑马型/低位防守）│
│ - 比赛级别判定（碾压局/实力差距/旗鼓相当）         │
│ - 风险因素识别                                   │
│ - 超级巨星破局分析                               │
│ - 模型一致度 + 平局信号检测                       │
└──────────────────────────────────────────────────┘
```

## 二、完整流程

### 步骤 1: 数据准备

```bash
# 赛程数据位置
/root/.openclaw/workspace/wc26-predict/extracted/worldcup-clean/2026-world-cup-predictor/data/match_cache.json

# Elo 数据库
strategy_engine.py → ELO_DB 字典（70 队）
```

**检查清单：**
- [ ] 赛程时间是否正确？(跨时区验证，ET→北京时间+12h)
- [ ] 新球队 Elo 是否在 ELO_DB 中？
- [ ] 已完成比赛比分是否已录入？

### 步骤 2: 运行预测

```python
from predict_server import generate_all_predictions
generate_all_predictions()
```

或在 API 上直接请求：
```bash
curl -X POST http://127.0.0.1:8080/api/refresh
```

**这会生成每个比赛的以下数据：**
- `winDrawLoss` — 各模型 + 集成的 H/D/A 概率
- `scoreProb` — Top 12 最可能比分 + 各模型概率
- `recommendation` — 推荐结果、比分、置信度
- `analysis` — 深度分析（风格/对位/风险/巨星因素）
- `reasoning` — 6 维度推理（实力/一致度/概率/风格/平局/大小球）
- `weights` — 自适应集成权重

### 步骤 3: 验证预测质量

按 README 规则校验：

| 规则 | 条件 | 动作 |
|------|------|------|
| Elo 差 > 250 | 碾压局 | 模型可信度高 |
| Elo 差 < 150 | 旗鼓相当 | 不碰胜平负 |
| 最高概率 ≥ 65% | 强信号 | 优先出手 |
| 最高概率 ≥ 55% | 可出手 | 考虑 |
| 最高概率 < 50% | 猜硬币 | 不推荐 |
| federico1809 平局 ≥ 35% + Elo差<150 | 平局信号 | 防范平局 |
| EV < 1% | 负期望值 | 不推荐 |

### 步骤 4: 部署前端

```bash
cp /root/.openclaw/workspace/laoye/worldcup-prediction.html /var/www/laoyeah/predict/index.html
```

### 步骤 5: 提供分析

**按 Football Match Deep Analysis Skill 框架输出分析报告：**

```
模块一：球队整体画像
  - 排名与积分形势
  - 近期比赛状态（含 Elo 分析和本届小组赛结果）
  - 战术体系与比赛风格（分类：超级巨星型/体系型/黑马型/低位防守）
  - 量化数据对比（Elo、概率、模型一致度）
  - 阵容与伤停（如可获取）

模块二：关键对位与战术细节
  - 核心对位分析
  - 中场控制权（基于模型概率推断）
  - 定位球（基于 Elo 差推断）
  - 双方弱点与突破口
  - 教练博弈（基于比赛风格推断）

模块三：模型量化分析
  - 5 模型概率对比（Hicruben → mikobinbin → AndyDu → amir42 → federico1809）
  - 集成概率
  - 模型一致度（H/D/A 投票）
  - 自适应权重分析
  - 比分概率 Top 5
  - 期望总进球

模块四：决策判断
  - 可出手性检查（Elo 差 > 150？概率 > 55%？）
  - 风险因素
  - 冷门路径
  - 置信度
  - 一句话总结
```

## 三、5 模型特性速查

| 模型 | 类型 | 可信度 | 用法 |
|------|------|:------:|------|
| **Hicruben** (v2) | Dixon-Coles Poisson | ⭐⭐⭐⭐ | 优先看，概率校准最优 |
| **mikobinbin** (v2) | 调整 Elo + Poisson | ⭐⭐⭐ | 弱队预警，给弱队更多机会 |
| **AndyDu** (v2) | 混合 Poisson/logistic | ⭐⭐⭐ | 概率极端化（慎用） |
| **amir42** | XGBoost | ⭐⭐⭐ | 机器学习，需 joblib |
| **federico1809** | XGBoost+聚类 | ⭐⭐(平局专用) | **唯一能预测平局的模型** |

## 四、自适应集成权重

| Elo 差 | Hicruben | mikobinbin | AndyDu | amir42 | federico1809 |
|:------:|:--------:|:----------:|:------:|:------:|:-----------:|
| < 150 | 0.20 | 0.20 | 0.20 | 0.20 | **0.20** |
| 150-250 | 0.21 | 0.21 | 0.21 | 0.21 | **0.16** |
| > 250 | 0.23 | 0.23 | 0.23 | 0.23 | **0.08** |

## 五、前端字段参考

| API 字段 | 前端显示 | 类型 |
|----------|---------|------|
| winDrawLoss.{model}| WDL 概率表格 | table |
| scoreProb[].avg | 比分概率 Top 12 | table |
| recommendation.result | 推荐结果 | header |
| recommendation.confidence | 置信度 | badge |
| analysis.matchLevel | 比赛级别 | badge |
| analysis.teamHome.style | 主队风格 | text |
| analysis.riskFactors | 风险列表 | list |
| analysis.drawSignal | 平局信号 | badge |
| reasoning[].label | 推理条目 | cards |

## 六、已知陷阱

1. **不要用 Poisson 模型预测平局** — 它们永远给 H 或 A 更高概率
2. **不要忽略 federico1809 的平局信号** — 唯一可用的平局检测
3. **不要在 Elo < 150 时重注胜平负**
4. **不要在小组赛最后一轮用模型** — 战意/轮换不可量化
5. **所有比分概率都很低（通常 <15%）** — 这是 Poisson 模型特性，正常
6. **数据源必须 cross-check** — 单一来源可能有误（见土耳其 vs 巴拉圭事件）

## 七、文件索引

| 文件 | 用途 |
|------|------|
| `predict_server.py` | FastAPI 服务器 + 深度分析生成 |
| `extracted/worldcup-clean/optimized_engine.py` | v2 预测引擎 |
| `extracted/worldcup-clean/strategy_engine.py` | v1 引擎（可回退） |
| `extracted/worldcup-clean/README.md` | 量化决策规则 |
| `data/predictions.json` | 缓存预测数据 |
| `laoye/worldcup-prediction.html` | 前端源码 |
| `SOP.md` | **本文档** |

## 八、快速启动

```bash
# 重新生成所有预测
curl -X POST https://laoyeah.win/api/refresh

# 检查状态
curl https://laoyeah.win/api/status

# 部署前端
cp ./laoye/worldcup-prediction.html /var/www/laoyeah/predict/index.html

# 查看预测
curl https://laoyeah.win/api/predictions?date=2026-06-20 | python3 -m json.tool
```
