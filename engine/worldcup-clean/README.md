# 世界杯预测 & 投注指南

> 写给未来的 Claude：每次用户问比赛预测或投注建议时，按这个流程来。

---

## 一、快速预测流程

### 1. 拿到赛程后

```bash
cd D:/HUAWEI/Desktop/claudecode体验/worldcup
PYTHONIOENCODING=utf-8 python strategy_engine.py --matches "TeamA,TeamB" "TeamC,TeamD"
```

需要单独跑 5 模型概率（不拉赔率）时，改 `run_predictions.py` 里的 MATCHES 列表执行。

### 2. 检查 Elo 数据

所有 Elo 存在 `strategy_engine.py` 的 `ELO_DB` 字典中。新球队不在里面的话，去 `world-cup-2026-prediction-model/data/elo-calibrated.json` 查，或者用默认 1700。

### 3. 拉实时赔率

```python
from strategy_engine import fetch_match_odds
odds = fetch_match_odds("TeamA", "TeamB", bookmaker="Pinnacle")
```

API key 在 `strategy_engine.py` 第 27 行，或环境变量 `ODDS_API_KEY`。每天 500 次免费请求，省着用。

---

## 二、5 模型特性（必须记住）

| 模型 | 类型 | 强项 | 弱项 | 可信度 |
|:---|:---|:---|:---|:---|
| **Hicruben** | Elo + Dixon-Coles Poisson | 概率校准最好（RPS 最低），强弱分明比赛准 | 平局预测 0% | ⭐⭐⭐⭐ |
| **mikobinbin** | 调整 Elo + Poisson | 给弱队更多机会，冷门预警 | 平局预测 0% | ⭐⭐⭐ |
| **AndyDu** | Elo + 混合 Poisson/logistic | 概率极端化（强队给更高%） | 平局预测 0%，LogLoss 最差 | ⭐⭐⭐ |
| **amir42** | XGBoost | 机器学习，捕捉非线性关系 | 平局预测 0%，需要 joblib | ⭐⭐⭐ |
| **federico1809** | XGBoost + 聚类特征 | **唯一能预测平局的模型**（42.9% 召回） | 整体准确率低（21.4%），倾向预测平局 | ⭐⭐（平局专用） |

### 2026 世界杯回测数据（14 场，06-11 ~ 06-16）

| 指标 | Hicruben | mikobinbin | AndyDu | amir42 | federico1809 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 准确率 | 42.9% | 42.9% | 42.9% | 42.9% | 21.4% |
| RPS | **0.179** | 0.181 | 0.196 | 0.183 | 0.194 |
| LogLoss | **1.047** | 1.085 | 1.237 | 1.089 | 1.122 |
| 平局召回 | 0% | 0% | 0% | 0% | **42.9%** |

**注意：本届前 14 场平局率 50%（7/14），远高于历史平均 25-28%。模型表现可能随赛事推进回归正常。**

---

## 三、决策规则（核心）

### 规则 1：只在高置信度时出手

```
IF 模型给某结果概率 >= 55%:
    → 可以考虑投注该结果
IF 模型给某结果概率 >= 65%:
    → 强信号，优先考虑
ELSE:
    → 不出手
```

回测显示模型在概率 >50% 时准确率约 69%（Hicruben 历史数据）。低于 55% 的预测本质上是猜硬币。

### 规则 2：Elo 差是第一过滤器

| Elo 差 | 含义 | 策略 |
|:---|:---|:---|
| > 250 | 碾压局 | 模型可信度高，可以考虑让球/大球 |
| 150-250 | 有差距但有悬念 | 看赔率 EV，谨慎出手 |
| < 150 | 实力接近 | **不碰胜平负**，除非有赔率套利 |

### 规则 3：平局怎么办

**Poisson 模型（Hicruben/mikobinbin/AndyDu/amir42）不能用来预测平局。** 回测证明的铁律。

判断平局的方法：
1. 看 federico1809 是否给了 >= 35% 平局概率
2. 如果 federico1809 预测平局，且 Elo 差 < 150 → 平局有一定可能性
3. **但不要单独依赖 federico1809 做投注决策**

**最安全的做法：实力接近的比赛（Elo 差 < 150），不碰胜平负，只看大小球。**

### 规则 4：大小球比胜平负更可靠

大小球不依赖平局预测，只依赖进球数期望值。

```
IF 模型期望总进球 >= 2.5 且赔率隐含概率 < 模型概率:
    → 考虑买大球
IF 模型期望总进球 <= 1.8 且赔率隐含概率 < 模型概率:
    → 考虑买小球
```

### 规则 5：EV > 1% 才考虑

```python
from strategy_engine import ev, kelly
ev_val = ev(model_prob, odds)
if ev_val > 0.01:  # 至少 1% 期望值
    kf = kelly(model_prob, odds)
    bet_fraction = min(kf * 0.5, 0.10)  # Half-Kelly，单注上限 10%
```

---

## 四、投注执行流程

### 每场比赛的标准流程

```
1. 拿到赛程
2. 查 Elo（ELO_DB 或 elo-calibrated.json）
3. 跑 5 模型 → 得到 H/D/A 概率 + 期望进球
4. 拉 Pinnacle 赔率
5. 计算每个市场的 EV
6. 应用决策规则：
   a. Elo 差 < 150？→ 跳过胜平负
   b. 最高概率 < 55%？→ 跳过
   c. EV < 1%？→ 跳过
   d. 通过所有过滤 → 计算 Kelly 仓位
7. 输出投注建议
```

### 仓位管理

| 参数 | 值 | 说明 |
|:---|:---|:---|
| Kelly 分数 | 0.5 | Half-Kelly，降低波动 |
| 单注上限 | 10% | 任何单注不超过总资金 10% |
| 串关上限 | 5% | 串关风险更高，更保守 |
| 总暴露上限 | 30% | 所有未结算投注总和 |
| 最低 EV | 1% | 低于此不投 |

### 串关规则

- 只用不同比赛的不同市场串
- 最多 3 串 1
- 串关用 1/4 Kelly（比单注更保守）
- 只串 EV > 3% 的选项

---

## 五、输出模板

用户问预测时，按以下格式输出（参考 `wc_report_format.md`）：

```
## 胜平负概率表
| 模型 | [主队]胜 | 平局 | [客队]胜 |
（顺序：AndyDu → amir42 → Hicruben → mikobinbin → federico1809）
| 5模型平均 | ... | ... | ... |

## 比分概率表（Top10）
| 比分 | Hicruben | mikobinbin | AndyDu | amir42 | federico1809 | 平均 |
（顺序：Hicruben → mikobinbin → AndyDu → amir42 → federico1809）
（按平均概率降序，取 Top10）

## 推荐
- 推荐结果：XXX（概率 XX%）
- 推荐比分：X-X（概率 XX%）
```

---

## 六、已知陷阱

### 不要做的事

1. **不要用 Poisson 模型预测平局** — 它们永远给 H 或 A 更高概率
2. **不要在 Elo 差 < 150 时重注胜平负** — 模型在这种比赛上准确率很低
3. **不要过度信任 AndyDu** — 它的 LogLoss 最差，概率极端化
4. **不要忽略 federico1809 的平局信号** — 虽然它整体差，但平局检测是唯一可用的
5. **不要串关超过 3 场** — 方差太大
6. **不要在小组赛最后一轮用模型** — 战意/轮换因素模型无法捕捉

### 要做的事

1. **优先看 Hicruben** — 概率质量最好
2. **大小球优先于胜平负** — 不依赖平局预测
3. **强弱分明的比赛出手** — Elo > 200 时模型可信度最高
4. **用 Pinnacle 赔率** — 最 sharp，返还率最高
5. **记录每场预测和结果** — 持续校准模型

---

## 七、文件索引

| 文件 | 用途 |
|:---|:---|
| `strategy_engine.py` | 主引擎：5 模型 + 赔率 + Kelly + 串关 |
| `optimized_engine.py` | v2 优化版模型（DC_RHO 调参 + 校准层） |
| `run_predictions.py` | 跑指定比赛的 5 模型概率 |
| `backtest_all5.py` | 5 模型回测 |
| `backtest_v1_vs_v2.py` | v1 vs v2 对比 |
| `world-cup-2026-prediction-model/data/elo-calibrated.json` | 校准后 Elo 数据 |
| `amir42-predictor/` | amir42 XGBoost 模型文件 |
| `federico1809-predictor/` | federico1809 XGBoost 模型文件 |

---

## 八、一句话总结

**用 Hicruben 看方向，用 federico1809 看平局信号，用 Pinnacle 赔率算 EV，只在 Elo > 200 且 EV > 1% 时出手，Half-Kelly 控制仓位。看不懂的比赛不碰。**
