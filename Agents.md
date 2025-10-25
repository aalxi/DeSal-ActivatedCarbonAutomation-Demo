Absolutely. Here’s the **entire PRD** as a single clean `.md` file — fully self-contained, properly formatted, and immediately usable as `README.md` or `SPEC.md` in a Cursor workspace.

---

# PAC Automation: Bloom Week Simulation  

**Purpose:**  
A visual, narrative demonstration of a rule-based PAC (Powdered Activated Carbon) dosing controller responding to real-world sensor signals over a “bloom week.”  
The demo shows detection, response, and stabilization with clear, explainable safety logic — rigorous enough for desalination experts yet visually intuitive for investors and judges.

---

## 1. Overview & Narrative Flow

**Duration:** ~30 seconds total  
**Experience:** The viewer watches an animated week-long simulation compressed into ~30 seconds. Data curves evolve as minimal captions narrate key moments.

### Narrative Sequence

| Time (s) | Caption | Visual Focus |
|-----------|----------|---------------|
| 0–3 | “Sensors online. Baseline stable.” | All boxes idle. Graph shows calm baseline. |
| 3–7 | “Day 2: Algae signals rising.” | Chl-a and FDOM begin to climb; “Sensor” box lights. |
| 7–12 | “Bloom detected. Controller activates.” | Bloom mode shading appears. “Logic” box pulses. PAC line ramps smoothly. |
| 12–17 | “Dosing adjusts with rate limits.” | “Dose” box lights; curve rises with smooth slope. |
| 17–22 | “Guardrails active: turbidity spike detected.” | Red alert marker; “Safety” box pulses amber; caption above graph. |
| 22–28 | “Bloom subsides. Controller returns to baseline.” | Signals fall; shaded region fades; PAC lowers gradually. |
| 28–30 | “Outcome: RO feedwater cleaner, system stable.” | Lower UV254 curve shown; boxes idle; calm state. |

---

## 2. Visual System

### Layout
- **Top Row (Process Flow):**  
  `SENSORS → LOGIC → DOSE → SAFETY`  
  - Boxes highlight/pulse when active.  
  - Thin connecting arrows with color echoes to graph lines.

- **Main Graph (Lower Area):**  
  - Two panes stacked vertically.

#### Upper Pane — Sensor Signals
- **Lines:**
  - Chl-a → coral orange  
  - FDOM → soft magenta-purple  
  - UV254 → aqua blue  
  - Turbidity/SDI → muted gray-green
- **Shaded Region:** translucent peach–aqua gradient when `bloom_mode = ON`.
- **Dashed Lines:** entry/exit thresholds shown horizontally.

#### Lower Pane — Dosing & Outcomes
- PAC setpoint → thick coral-red line  
- PAC base → dotted line  
- PAC max → thin dashed line  
- RO-influent UV254 → thin light-blue line (decreasing with dosing)  
- Alert markers:
  - ▲ amber = guardrail  
  - ✖ red = sensor fault

### Motion
- Curves animate smoothly (cubic easing).  
- Bloom shading fades in/out (0.5 s).  
- Box highlights pulse softly, no flashing.  
- Captions fade in/out top-center.

### Palette (Warm-Natural)
- Background: off-white with warm tint  
- Lines: pastel coral/orange/magenta/aqua  
- Shading: low-opacity gradients (no solid fills)  
- Text: dark gray for clarity

---

## 3. Interactive Controls

**Control Panel (right or below graph):**
- Sliders (numeric readout):
  - `k1` (UV weight)
  - `k2` (FDOM weight)
  - `k3` (Chl bias)
  - `PAC_max`
  - Mode thresholds (Chl_enter/exit, UV_enter/exit, FDOM_enter/exit)
- Instruction text:  
  _“Adjust values, then press **Run Simulation** to replay.”_
- **Run Simulation** button:
  - Recomputes all signals and controller logic.
  - Replays full 30s animation sequence.
- Inputs are validated and bounded — simulation can’t break.

### Default Constants
```text
PAC_base = 1 mg/L
PAC_min = PAC_base
PAC_max = 20 mg/L
max_delta_per_hour = 3 mg/L
k1 = 100
k2 = 0.03
k3 = 0.15
Chl_enter = 8 µg/L, Chl_exit = 5 µg/L
UV_enter = 0.06 1/cm, UV_exit = 0.05 1/cm
FDOM_enter = 80 RFU, FDOM_exit = 60 RFU
````

---

## 4. Data & Logic Layer

### Data Shape

15-min resolution over 8 simulated days.

```
timestamp, uv254_1_per_cm, fdom_rfu, chlorophyll_ug_per_l,
turbidity_ntu, sdi_index, mode_bloom (0/1),
pac_setpoint_mg_per_l, ro_influent_uv254,
alert_flag (0/1), sensor_fault (0/1)
```

### Synthetic Data Generator

* Simulates calm → bloom → calm progression.
* Adds Gaussian noise to signals.
* FDOM and UV254 lag behind Chl-a.
* Turbidity spikes briefly during bloom.
* RO-influent UV254 = feed UV254 × (1 − 0.02 × PAC_setpoint) + noise.

### Controller Logic

```text
IF (Chl-a >= Chl_enter) OR (UV254 >= UV_enter AND FDOM >= FDOM_enter)
  → mode = bloom
ELSE IF (Chl-a <= Chl_exit AND UV254 <= UV_exit AND FDOM <= FDOM_exit)
  → mode = normal
```

In **normal mode:**

```
PAC_setpoint = PAC_base
```

In **bloom mode:**

```
PAC_target = PAC_base + k1*UV254 + k2*FDOM + k3*max(0, Chl-a - Chl_exit)
PAC_clamped = clamp(PAC_target, PAC_min, PAC_max)
PAC_rate_limited = limit_change(prev_PAC, PAC_clamped, ±3 mg/L per hr)
IF (UV254 ↓ while Turbidity ↑ rapidly) THEN
  hold PAC_rate_limited constant
  raise alert_flag = 1
```

Sensor fault logic:

```
IF any sensor missing > 30 min:
  PAC_setpoint = PAC_base
  sensor_fault = 1
```

---

## 5. Alerts & Overlay System

### Alert Behavior

* **Amber ▲:** turbidity/guardrail alert

  * “SAFETY” box glows amber for ~2 s.
  * Caption: *“Guardrail active: holding dose.”*
* **Red ✖:** sensor fault

  * “SAFETY” box glows red for ~2 s.
  * Caption: *“Sensor fault: reverting to base.”*
* Icons appear at event timestamp on graph; brief tooltip shows reason.

### Caption System

* Central overlay (top-center)
* Text transitions fade (no scroll)
* Follows timing sequence in Section 1

---

## 6. Aesthetic & Technical Notes

**Tone:** neutral, professional — “research demo,” not hype.
**Font:** sans-serif, medium weight (e.g., Inter, Roboto).
**Animation speed:** calm, smooth transitions.
**Color saturation:** ~70% max; never neon.
**Overall feel:** precise, stable, and ocean-inspired.

### Optional Future Add-ons

* Hover tooltips showing live numeric values.
* Toggle for numeric dashboard view.
* Button to compare two simulation runs.
* PNG export for presentations.

---

## Deliverable Summary

A single-page interactive demo (Jupyter or Streamlit) showing:

* Weeklong bloom simulation
* Sensor signal evolution
* PAC dose adaptation with safety logic
* Visual and textual storytelling for clarity
* Adjustable coefficients and thresholds with one-click re-simulation
* Polished, explainable visuals acceptable to desalination experts

---

**End of PRD**

```

---

✅ This `.md` file is complete, precise, and production-ready.  
Cursor or any LLM-based build agent can directly scaffold:
- synthetic data generation  
- interactive plot  
- UI elements  
- animation and caption logic  
— all without additional clarification.
```
