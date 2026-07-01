"""
AegisDrive MRM demo entry point.

CLI fallback:
    python app.py --list
    python app.py --scenario S4
    python app.py --all

Streamlit UI:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

try:
    from .decision_engine import decide_minimum_risk, format_decision
    from .risk_state import RiskState, find_scenario, load_scenarios, percent
    from .paper_reproduction import run_paper_world_model
    from .world_model_mock import run_world_model_mock
except ImportError:
    from decision_engine import decide_minimum_risk, format_decision
    from risk_state import RiskState, find_scenario, load_scenarios, percent
    from paper_reproduction import run_paper_world_model
    from world_model_mock import run_world_model_mock


def _scenario_path() -> Path:
    return Path(__file__).with_name("scenarios.json")


def select_world_model(state: RiskState, decision, model_kind: str = "mock"):
    """Run the selected demo world-model helper."""
    if model_kind == "paper":
        return run_paper_world_model(state, decision)
    return run_world_model_mock(state, decision)


def run_pipeline(state: RiskState, model_kind: str = "mock") -> Dict[str, object]:
    decision = decide_minimum_risk(state)
    world_model = select_world_model(state, decision, model_kind=model_kind)
    return {
        "scenario": state.to_dict(),
        "decision": decision.to_dict(),
        "world_model_kind": model_kind,
        "world_model": world_model.to_dict(),
    }


def _print_prediction_table(world_model_result: Dict[str, object]) -> None:
    print("\n[世界模型辅助：候选策略 1/3/5s 风险推演]")
    print("-" * 104)
    print(f"{'状态':<8} {'候选策略':<24} {'1s':>6} {'3s':>6} {'5s':>6} {'评分':>7}  解释")
    print("-" * 104)
    for pred in world_model_result["predictions"]:
        risks = {p["horizon_s"]: p["risk"] for p in pred["future_risks"]}
        label = pred["label"][:22]
        print(
            f"{pred['status']:<8} {label:<24} "
            f"{risks.get(1, 0):>6.2f} {risks.get(3, 0):>6.2f} {risks.get(5, 0):>6.2f} "
            f"{pred['score']:>7.2f}  {pred['reason']}"
        )
    print("-" * 104)


def _print_cli_result(state: RiskState, as_json: bool = False, model_kind: str = "mock") -> None:
    result = run_pipeline(state, model_kind=model_kind)
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    decision = result["decision"]
    assessment = decision["assessment"]
    world_model = result["world_model"]

    print("\n" + "=" * 96)
    print(f"{state.scenario_id} | {state.name}")
    print(state.description)
    print("=" * 96)
    print(format_decision(decide_minimum_risk(state)))
    print("\n[候选策略约束]")
    for action in decision["candidate_actions"]:
        flag = "允许" if action["allowed"] else "禁止"
        print(f"- {flag:<2} | {action['label']}：{action['reason']}")
    _print_prediction_table(world_model)
    print(f"世界模型({model_kind})推荐: {world_model['recommended_label']} [{world_model['recommended_action_id']}]")
    print(world_model["explanation"])
    if state.expected_strategy:
        print(f"场景预期策略: {state.expected_strategy}")
    print(
        f"风险快照: TRS={assessment['takeover_score']}/3, "
        f"Road={assessment['road_risk_score']:.2f}, "
        f"System={assessment['system_risk_score']:.2f}, "
        f"Fused={assessment['fused_risk_score']:.2f}"
    )


def run_cli(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AegisDrive 最小风险决策 Demo")
    parser.add_argument("--scenario", "-s", default="S4", help="场景 ID，例如 S1/S4/S6")
    parser.add_argument("--all", action="store_true", help="依次运行全部场景")
    parser.add_argument("--list", action="store_true", help="列出场景库")
    parser.add_argument("--json", action="store_true", help="输出结构化 JSON")
    parser.add_argument(
        "--world-model",
        choices=["mock", "paper"],
        default="mock",
        help="选择世界模型辅助模块",
    )
    args = parser.parse_args(argv)

    scenarios = load_scenarios(_scenario_path())
    if args.list:
        print("可用场景：")
        for scenario in scenarios:
            print(f"- {scenario.scenario_id}: {scenario.name} | 预期: {scenario.expected_strategy}")
        return 0

    if args.all:
        for scenario in scenarios:
            _print_cli_result(scenario, as_json=args.json, model_kind=args.world_model)
        return 0

    scenario = find_scenario(scenarios, args.scenario)
    _print_cli_result(scenario, as_json=args.json, model_kind=args.world_model)
    return 0


def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def render_streamlit_app() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="AegisDrive MRM Demo", page_icon="🚗", layout="wide")
    scenarios = load_scenarios(_scenario_path())
    scenario_map = {f"{s.scenario_id} - {s.name}": s for s in scenarios}

    st.title("AegisDrive 最小风险决策 Demo")
    st.caption("结构化 RiskState → 接管能力判断 → 非世界模型 baseline → 世界模型辅助 1/3/5s 风险推演")

    with st.sidebar:
        st.header("演示控制台")
        selected_key = st.selectbox("选择预设场景", list(scenario_map.keys()), index=3)
        model_kind = st.radio("世界模型模式", ["mock", "paper"], index=0, horizontal=True)
        show_json = st.checkbox("显示完整 RiskState JSON", value=True)
        st.caption(f"当前使用：{model_kind}")
        st.markdown("---")
        st.markdown("**三人组对应任务**")
        st.markdown("A：RiskState + 状态机 baseline")
        st.markdown("B：世界模型辅助预测 mock")
        st.markdown("C：场景库 + 测试与汇报素材")
        st.markdown("---")
        st.info("本 demo 不接摄像头、不控制真实底盘，只验证最小风险决策流程。")

    state = scenario_map[selected_key]
    decision = decide_minimum_risk(state)
    world_model = select_world_model(state, decision, model_kind=model_kind)
    assessment = decision.assessment

    st.subheader(f"{state.scenario_id} | {state.name}")
    st.write(state.description)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("接管准备度 TRS", f"{assessment.takeover_score}/3", assessment.takeover_label)
    col2.metric("道路风险", f"{assessment.road_risk_score:.2f}", assessment.road_risk_label)
    col3.metric("系统风险", f"{assessment.system_risk_score:.2f}", percent(assessment.system_risk_score))
    col4.metric("综合风险", f"{assessment.fused_risk_score:.2f}", assessment.fused_risk_label)

    tab_state, tab_baseline, tab_world, tab_report = st.tabs([
        "RiskState 输入",
        "非世界模型 baseline",
        "世界模型辅助预测",
        "汇报口径",
    ])

    with tab_state:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### 驾驶员状态")
            st.json(state.driver.__dict__)
            st.markdown("#### 接管判断理由")
            for reason in assessment.takeover_reasons:
                st.write("- " + reason)
        with c2:
            st.markdown("#### 道路与系统状态")
            st.json({"road": state.road.__dict__, "system": state.system.__dict__})
            st.markdown("#### 风险理由")
            for reason in assessment.risk_reasons:
                st.write("- " + reason)
        if show_json:
            st.markdown("#### 完整结构化输入")
            st.json(state.to_dict())

    with tab_baseline:
        st.markdown("#### 状态机输出")
        b1, b2, b3 = st.columns(3)
        b1.metric("FSM 状态", decision.fsm_state_label, decision.fsm_state)
        b2.metric("Baseline 策略", decision.baseline_strategy_label, decision.baseline_strategy)
        b3.metric("紧急程度", decision.urgency)
        st.markdown("#### 决策解释")
        for reason in decision.reasons:
            st.write("- " + reason)

        st.markdown("#### 候选策略约束")
        candidate_df = pd.DataFrame([
            {
                "候选策略": a.label,
                "动作 ID": a.action_id,
                "是否允许": "允许" if a.allowed else "禁止",
                "约束理由": a.reason,
                "作用说明": a.effect_hint,
            }
            for a in decision.candidate_actions
        ])
        st.dataframe(candidate_df, use_container_width=True, hide_index=True)

    with tab_world:
        st.markdown("#### 推荐结论")
        st.success(f"推荐：{world_model.recommended_label} [{world_model.recommended_action_id}]")
        st.caption(f"当前世界模型模式：{model_kind}")
        st.write(world_model.explanation)

        pred_rows = []
        line_data = {"time_s": [1, 3, 5]}
        for pred in world_model.predictions:
            risk_by_h = {p.horizon_s: p.risk for p in pred.future_risks}
            pred_rows.append({
                "状态": pred.status,
                "候选策略": pred.label,
                "动作 ID": pred.action_id,
                "1s风险": risk_by_h.get(1),
                "3s风险": risk_by_h.get(3),
                "5s风险": risk_by_h.get(5),
                "评分": pred.score,
                "解释": pred.reason,
            })
            line_data[pred.label] = [risk_by_h.get(1), risk_by_h.get(3), risk_by_h.get(5)]

        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)
        st.markdown("#### 1/3/5 秒风险趋势")
        st.line_chart(pd.DataFrame(line_data).set_index("time_s"))
        st.caption("数值越低表示预测风险越低。mock 为规则预测；paper 为轻量论文风格复现，只做候选动作 rollout、风险代价评估和策略排序。")

    with tab_report:
        st.markdown("#### 阶段性汇报可讲口径")
        st.markdown(
            "我们三人组负责最小风险决策模块，不直接控制真实车辆。Demo 使用预设场景模拟舱内外感知输出，"
            "先由 RiskState 统一表达驾驶员接管准备度、道路风险和系统状态，再由规则状态机给出 baseline 策略。"
            "世界模型辅助模块对候选策略进行 1/3/5 秒风险推演；paper 模式是轻量论文风格复现，"
            "用于候选动作 rollout、风险代价评估和策略排序，不是训练型真实世界模型。"
        )
        st.markdown("#### 当前 demo 满足的最小交付物")
        st.write("- 5 个以上典型失效/接管场景")
        st.write("- RiskState 输入接口与接管准备度评分")
        st.write("- 非世界模型 MRM 状态机 baseline")
        st.write("- 世界模型辅助候选策略风险趋势")
        st.write("- Streamlit 网页演示 + 命令行备用演示")
        st.markdown("#### 下一步接入")
        st.code(
            "src.internal.* 输出疲劳/分心/闭眼/头姿 → RiskState.driver\n"
            "src.external.* 输出 TTC/距离/道路风险 → RiskState.road\n"
            "src.core.risk_fusion 输出综合风险/可信度 → RiskState.system 或 assessment",
            language="text",
        )


if __name__ == "__main__":
    if _running_inside_streamlit():
        render_streamlit_app()
    else:
        raise SystemExit(run_cli())
