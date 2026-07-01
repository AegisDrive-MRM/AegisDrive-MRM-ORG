# Lightweight world-model reproduction module

This folder implements a minimal, runnable reproduction of the core idea behind
world-model-assisted autonomous-driving planning papers in IEEE T-ITS style:

1. represent the current scene as a compact driving-aware state;
2. enumerate candidate MRM actions;
3. roll out each action for 1/3/5 seconds;
4. compute a risk-aware cost from TTC, distance, driver readiness, perception
   uncertainty, lateral maneuver risk, rear risk and comfort cost;
5. choose the lowest-cost MRM action.

This is not a full reproduction of CARLA, RL, MPC or MPPI training. It is a
stage-report reproduction of the decision mechanism: candidate-action rollout
plus future risk evaluation.

## Run

From `mrm_demo/`:

```bash
python -m paper_reproduction.reproduce_demo --scenario S4
python -m paper_reproduction.reproduce_demo --all
```

Or use the integrated app:

```bash
python app.py --scenario S4 --world-model paper
python app.py --all --world-model paper
streamlit run app.py
```

## Files

- `risk_cost.py`: compact state/action model and risk cost function.
- `world_model_predictor.py`: constant-acceleration rollout for 1/3/5 s.
- `evaluate_candidates.py`: candidate scoring and recommendation.
- `reproduce_demo.py`: standalone command-line reproduction demo.

## How to explain in the report

The non-world-model baseline uses rules on the current frame. The lightweight
world model predicts how each candidate action may change future risk, then
selects the action with the lowest weighted future-risk cost. This makes the MRM
module more predictive and more explainable without requiring a heavy simulator
or GPU training pipeline at the current stage.
