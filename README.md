# PAC Automation: Bloom Week Simulation

This repository implements the interactive demo described in `Agents.md`. The Streamlit
application generates synthetic bloom-week sensor data, applies the PAC dosing controller,
and narrates the eight-day sequence in a 30-second animation.

## Features
- Synthetic 15-minute resolution signals for chlorophyll-a, FDOM, UV254, turbidity, and SDI with controlled bloom progression.
- Rule-based controller implementing bloom detection, rate-limited dosing, clamping, and guardrail holds when turbidity spikes while UV254 falls.
- Narrative visualization with Plotly animation, dual-pane graph, bloom-mode shading, top-row process boxes, and caption overlays that follow the PRD timeline.
- Adjustable controller coefficients and thresholds via sidebar sliders, plus a **Run Simulation** button that regenerates the full animation with validated ranges.

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the demo:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Use the sidebar controls to tune weights, guardrails, and thresholds, then press **Run Simulation** to replay the narrative.

## Repository Structure
- `streamlit_app.py` – Streamlit entry point containing the data generator, controller logic, and visualization layout.
- `requirements.txt` – Python dependencies.
- `Agents.md` – Original PRD/specification that guided the implementation.

## Notes
- Random noise ensures each simulation run feels organic while remaining within safe, explainable bounds.
- Guardrail alerts display amber markers and highlight the **SAFETY** box when turbidity spikes during decreasing UV254 conditions.
