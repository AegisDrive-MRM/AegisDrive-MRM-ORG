# AegisDrive 最小风险决策 Demo

本目录是按照三人组阶段性汇报要求生成的独立 demo，放在 `IDMS_Project/mrm_demo/` 下，不修改原有 `main.py`、舱内/舱外检测模块或多模态融合主程序。

## 1. Demo 展示什么

Demo 使用预设 `RiskState` 场景模拟上游输出：驾驶员状态、道路风险、系统状态和感知可信度。系统会输出：

1. 驾驶员是否具备接管能力，形成 0-3 级 TRS（Takeover Readiness Score）。
2. 非世界模型 baseline 的 MRM 状态机决策。
3. 世界模型辅助模块对候选策略未来 1/3/5 秒风险变化的推演。
4. 推荐的最小风险策略及解释。

> 当前版本是演示原型，不接真实摄像头，不控制真实车辆底盘。`paper_reproduction/` 是轻量论文风格复现，只做候选动作 rollout、风险代价评估和策略排序，不是训练型真实世界模型。

## 2. 文件结构

```text
mrm_demo/
├── app.py                 # Streamlit 展示界面 + 命令行 fallback
├── scenarios.json         # 8 个预设测试场景
├── risk_state.py          # RiskState、接管准备度、道路风险、综合风险计算
├── decision_engine.py     # 非世界模型状态机 baseline 与策略输出
├── world_model_mock.py    # 世界模型辅助预测 mock 版
├── paper_reproduction/    # 轻量论文风格 world-model 复现 demo
├── live_adapter.py        # 可选 JSON runtime 快照适配器
├── README.md              # 运行方式与模块说明
├── requirements_mrm.txt   # Streamlit 网页演示依赖
├── sample_run.txt         # 全场景命令行测试输出样例
├── sample_run_paper.txt   # paper 模式全场景命令行输出样例
└── __init__.py
```

## 3. 快速运行

在项目根目录或本目录下均可运行。命令行版本不需要额外安装 Streamlit。

```bash
cd IDMS_Project/mrm_demo
python app.py --list
python app.py --scenario S4
python app.py --all
python app.py --scenario S4 --world-model paper
python app.py --all --world-model paper
python -m paper_reproduction.reproduce_demo --scenario S4
python -m paper_reproduction.reproduce_demo --all
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

负责 `world_model_mock.py` 和 `paper_reproduction/`。汇报时重点说明：

- 当前先用 mock 规则实现未来 1/3/5 秒风险趋势。
- paper 模式是轻量论文风格复现，体现候选动作 rollout + 风险代价评估 + 策略排序。
- 输入为同一个 RiskState 和候选策略集合。
- baseline 安全规则仍然优先，世界模型辅助结果只用于候选策略比较与解释。

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

## IDMS 输出接入 RiskState 适配层

`live_adapter.py` 负责把原 IDMS 的舱内、舱外、风险融合与系统状态输出转换为 `RiskState`，供 `decision_engine.decide_minimum_risk`、`world_model_mock.py` 和 `paper_reproduction/` 继续使用。

适配层支持 `dict`、dataclass 和普通对象输入。舱内输入可以来自 `face_mesh.py` 的 `face_data` 或 `driver_state.py` 的 `DriverState`；舱外输入可以是单个目标 `dict` 或 `list[dict]`，列表会优先选择 TTC 最小、其次距离最近的前向目标；融合和系统状态可以提供 `fused_score`、`fused_level`、`sensor_degraded`、`perception_confidence` 等字段。

缺失输入、无人脸、监测不可用、TTC 极低、系统退化都不会被默认为安全。无人脸不会被当作疲劳，但会被当作驾驶员接管不可依赖；TTC 极低和系统失效会推高道路/系统风险，并由 baseline 决策模块决定是否进入 MRM。

后续真实主程序可以通过 `build_risk_state_from_idms(...)` 接入：

```python
from mrm_demo.live_adapter import build_risk_state_from_idms
from mrm_demo.decision_engine import decide_minimum_risk

risk_state = build_risk_state_from_idms(
    face_data=face_data,
    driver_state=driver_state,
    vehicle_data=vehicle_data,
    fusion_result=fusion_result,
    system_status=system_status,
)
decision = decide_minimum_risk(risk_state)
```

当前 adapter 仍是接口层，不直接控制车辆，不替代 baseline，也不改变 `scenarios.json` 预设场景 demo 的运行方式。

## 7. 汇报口径示例

我们三人组负责最小风险决策模块。当前 demo 先不接真实摄像头和底盘，而是用结构化 RiskState 模拟舱内外感知结果。系统首先判断驾驶员是否可接管；如果可接管，则优先请求接管或增强提醒；如果不可接管，则进入最小风险处置状态机。与此同时，世界模型辅助模块会对候选策略进行未来 1/3/5 秒风险推演，帮助解释为什么选择紧急制动、车道内停车或路肩停车；paper 模式是轻量论文风格复现，不是训练型真实世界模型。
