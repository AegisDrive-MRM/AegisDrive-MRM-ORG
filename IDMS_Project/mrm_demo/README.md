# AegisDrive 最小风险决策 Demo

本目录是按照三人组阶段性汇报要求生成的独立 demo，放在 `IDMS_Project/mrm_demo/` 下，不修改原有 `main.py`、舱内/舱外检测模块或多模态融合主程序。

## 1. Demo 展示什么

Demo 使用预设 `RiskState` 场景模拟上游输出：驾驶员状态、道路风险、系统状态和感知可信度。系统会输出：

1. 驾驶员是否具备接管能力，形成 0-3 级 TRS（Takeover Readiness Score）。
2. 非世界模型 baseline 的 MRM 状态机决策。
3. 世界模型 mock 对候选策略未来 1/3/5 秒风险变化的推演。
4. 推荐的最小风险策略及解释。

> 当前版本是演示原型，不接真实摄像头，不控制真实车辆底盘。后续可把 `scenarios.json` 替换为真实 IDMS 模块输出。

## 2. 文件结构

```text
mrm_demo/
├── app.py                 # Streamlit 展示界面 + 命令行 fallback
├── scenarios.json         # 8 个预设测试场景
├── risk_state.py          # RiskState、接管准备度、道路风险、综合风险计算
├── decision_engine.py     # 非世界模型状态机 baseline 与策略输出
├── world_model_mock.py    # 世界模型辅助预测 mock 版
├── README.md              # 运行方式与模块说明
├── requirements_mrm.txt   # Streamlit 网页演示依赖
├── sample_run.txt         # 全场景命令行测试输出样例
└── __init__.py
```

## 3. 快速运行

在项目根目录或本目录下均可运行。命令行版本不需要额外安装 Streamlit。

```bash
cd IDMS_Project/mrm_demo
python app.py --list
python app.py --scenario S4
python app.py --all
```

网页演示版本：

```bash
cd IDMS_Project/mrm_demo
pip install -r requirements_mrm.txt
streamlit run app.py
```

建议阶段性汇报时使用 Streamlit 页面；若现场环境无法安装依赖，使用 `python app.py --scenario S4` 作为备用演示。

## 4. 预设场景

| 场景 | 内容 | 预期展示 |
|---|---|---|
| S1 | 正常低风险 | NORMAL / 继续监测 |
| S2 | 轻度分心 + 中等风险 | WARNING / 强提醒或请求接管 |
| S3 | 疲劳 + 高风险 | MRM 准备或强制减速 |
| S4 | 无响应 + 极高风险 | MRM_EXECUTE / 紧急制动 |
| S5 | 传感器退化 + 侧向信息不足 | 禁止变道，车道内安全停车 |
| S6 | 路肩可用 + 驾驶员不可接管 | 靠边/路肩停车 |
| S7 | ODD 退出 + 驾驶员可接管 | 优先请求驾驶员接管 |
| S8 | 前方急减速 + 驾驶员分心 | 强制减速保持车道 |

## 5. 三人组分工对应

### 成员 A：RiskState + baseline 状态机

负责 `risk_state.py` 与 `decision_engine.py`。汇报时重点说明：

- 不直接做车辆控制，而是做最小风险策略决策。
- 通过 TRS 判断驾驶员能否接管。
- 通过状态机输出 NORMAL、WARNING、TAKEOVER_REQUEST、MRM_PREPARE、MRM_EXECUTE、MRC_REACHED。

### 成员 B：世界模型辅助预测

负责 `world_model_mock.py`。汇报时重点说明：

- 当前先用 mock 规则实现未来 1/3/5 秒风险趋势。
- 输入为同一个 RiskState 和候选策略集合。
- 后续可以替换为轻量时序模型、树模型或真正的世界模型。

### 成员 C：场景库 + 测试与汇报素材

负责 `scenarios.json`、运行截图和测试记录。汇报时重点说明：

- 场景覆盖正常、分心、疲劳、无响应、传感器退化、路肩可用、ODD 退出等典型情况。
- 每个场景都有输入、预期策略、baseline 输出、世界模型 mock 输出。

## 6. 后续接入真实 IDMS 的接口方向

```text
src.internal.*   -> RiskState.driver
  - eye_closed_sec / perclos / yawn_frequency_min / distracted_sec / head_yaw_deg / head_pitch_deg / no_response_sec

src.external.*   -> RiskState.road
  - front_distance_m / min_ttc_sec / relative_speed_mps / road_risk_hint / lane_confidence / shoulder_available

src.core.*       -> RiskState.system 或 RiskAssessment
  - perception_confidence / sensor_degraded / odd_exit / system_failure / fused risk
```

## 7. 汇报口径示例

我们三人组负责最小风险决策模块。当前 demo 先不接真实摄像头和底盘，而是用结构化 RiskState 模拟舱内外感知结果。系统首先判断驾驶员是否可接管；如果可接管，则优先请求接管或增强提醒；如果不可接管，则进入最小风险处置状态机。与此同时，世界模型 mock 会对候选策略进行未来 1/3/5 秒风险推演，帮助解释为什么选择紧急制动、车道内停车或路肩停车。
