# Visual & Interface Design System (Design.md)

## 1. Design Philosophy & Visual Tone
The visual interface for the **Battery Fleet Intelligence Platform** is built on an **industrial dark-mode telemetry aesthetic**—reminiscent of mission-control energy consoles and modern high-tech automotive BMS displays.

### Key Aesthetic Principles
* **Deep Contrast & Low Eye Fatigue:** A true dark slate foundation (`#0b0f19`) reduces ocular strain during extended monitoring sessions while making colored status indicators pop with vibrant clarity.
* **Domain-Specific Color Semantics:** Colors are not decorative; they are strictly mapped to physical domains:
  - **Electrical ($V, I$):** Electric Cyan (`#00f0ff`)
  - **Thermal ($T, \dot{T}$):** Solar Amber / Crimson (`#ff9f1c` / `#ff3366`)
  - **Health ($SoC, SoH, RUL$):** Emerald Green (`#00e676`) to Coral Red (`#ff5252`)
* **Information Scannability:** High-density, glanceable metrics with secondary progressive disclosure for deep-dive electrochemical curves.
* **Micro-Animations & Dynamic Feedback:** Subtle glowing pulse animations on active real-time MQTT feeds and live alert indicators.

---

## 2. Color Palette & Design Tokens

```css
:root {
  /* Surface & Background Colors */
  --bg-app: #0b0f19;              /* Deep background slate */
  --bg-surface-primary: #121929;  /* Primary card / panel background */
  --bg-surface-secondary: #1a233a;/* Elevated card / hover state */
  --border-subtle: #23314f;       /* Subtle card borders */
  --border-highlight: #3b507d;    /* Active / focused card borders */

  /* Domain Color Tokens */
  --domain-electrical: #00f0ff;   /* Voltage, Current, Coulomb counting */
  --domain-thermal: #ff9f1c;      /* Temperature (°C), thermal gradients */
  --domain-thermal-alert: #ff3366;/* Over-temperature warning (> 45°C) */
  --domain-aging: #b388ff;        /* Capacity fade (Ah), cycle count */

  /* Battery State & Status Indicators */
  --status-healthy: #00e676;      /* SoH > 85%, normal operation */
  --status-warning: #ffd600;      /* SoH 75-85%, approaching EoL knee */
  --status-critical: #ff5252;     /* SoH < 75% or active thermal runaway risk */
  --status-relaxation: #00b0ff;   /* Electrochemical relaxation rebound event */

  /* Neutral Typography */
  --text-primary: #f0f4fc;        /* Primary headers and prominent values */
  --text-secondary: #94a3b8;      /* Labels, units, secondary descriptors */
  --text-muted: #64748b;          /* Timestamps, disabled elements */

  /* Functional Accents */
  --accent-cyan: #00f0ff;
  --accent-glow: rgba(0, 240, 255, 0.15);
}
```

---

## 3. Typography & Hierarchy

To balance readability with technical density, the interface pairs a modern geometric sans-serif for UI chrome with a clean monospace font for numeric readings and timestamps.

| Purpose | Font Family | Weight | Size | Letter Spacing |
| :--- | :--- | :--- | :--- | :--- |
| **Page / Asset Title** | `Inter`, -apple-system, sans-serif | 700 (Bold) | 24px / 1.5rem | -0.02em |
| **Section Header** | `Inter`, -apple-system, sans-serif | 600 (Semi-bold)| 16px / 1.0rem | -0.01em |
| **Large Metric (Gauge / KPI)**| `JetBrains Mono`, `Roboto Mono`, monospace | 700 (Bold) | 32px / 2.0rem | -0.03em |
| **Body & Card Text** | `Inter`, -apple-system, sans-serif | 400 (Regular) | 14px / 0.875rem| normal |
| **Telemetry Labels & Units**| `Inter`, -apple-system, sans-serif | 500 (Medium) | 12px / 0.75rem | 0.05em (Caps) |
| **Timestamps & Serial Logs**| `JetBrains Mono`, monospace | 400 (Regular) | 11px / 0.6875rem| 0.02em |

Google Fonts Import:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
```

---

## 4. Layout Architecture & Component Hierarchy

The dashboard layout utilizes a responsive CSS Grid system structured into 4 key visual zones:

```
+--------------------------------------------------------------------------------------------------+
| HEADER: Fleet Status | Ingest Rate | Asset Selector (Asset #1 - #10) | Cloud Budget Watch ($0.00)|
+--------------------------------------------------------------------------------------------------+
| QUICK KPI STRIP:                                                                                 |
| [ State of Charge ]    [ State of Health ]    [ Remaining Useful Life ]    [ Max Cell Temp ]     |
|       84.2%                   91.6%                 142 Cycles                  32.4°C           |
| (Dynamic SVG Gauge)    (Nominal: 2.0 Ah)       (Threshold: 75% EoL)        (Status: Normal)      |
+--------------------------------------------------------------------------------------------------+
| MULTI-DOMAIN SYNCHRONIZED TIMELINE (Plotly.js):                                                  |
| ── Voltage (V) & Current (A) [Left Axis]                                                         |
| ── Cell Surface Temperature (°C) [Right Axis]                                                    |
| ── Preceding Rest Duration ($t_{rest}$) & Relaxation Markers (Blue Dotted Flags)                 |
+--------------------------------------------------------------------------------------------------+
| DEEP DIVE ANALYTICS GRID (2 Columns):                                                            |
| COLUMN 1:                                         COLUMN 2:                                      |
| Capacity Fade Trajectory (Cycles 1 - N)           Differential Capacity Spectrum (dQ/dV)         |
| ── Actual NASA Measured Capacity                  ── Identified Redox Phase Peaks                |
| ── PI-MDCNet Hybrid Model (Trend-Preserved)       ── Peak Shift vs Aging Benchmark               |
| ── Baseline Comparison Overlays (XGB/LSTM)                                                       |
+--------------------------------------------------------------------------------------------------+
| FOOTER / SYSTEM LOG: Event stream, API Gateway latency (< 25ms), IoT Connection Status (Active) |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. Key Interactive Components

### 5.1 Real-Time SVG Radial Gauges
* Circular progress gauge for **SoC** with an animated stroke dash offset.
* Dynamic stroke color transitioning from emerald (`#00e676`) at > 40%, yellow (`#ffd600`) at 20–40%, and red (`#ff5252`) at < 20%.

### 5.2 Interactive Synchronized Telemetry Plots (Plotly.js)
* Dual y-axes for Voltage and Temperature.
* Crosshair cursor scrubbing across time updates the instantaneous readouts synchronously.
* **Rest Relaxation Callouts:** Visual vertical annotations marking whenever an asset finishes an extended rest period, visually validating that the model respects electrochemical relaxation.

### 5.3 Differential Capacity ($dQ/dV$) Curve Inspector
* Enables researchers and faculty to visually verify the $dQ/dV$ feature peaks extracted via Savitzky-Golay filtering during charging cycles.

### 5.4 Closed-Loop Emergency Control Panel
* Visual toggle allowing simulated operator intervention (sending an MQTT control message to throttle maximum charging C-rate if temperature exceeds 40°C).

---

## 6. Accessibility & Responsiveness
* **Contrast Compliance:** All text-to-background combinations meet or exceed WCAG 2.1 AA requirements (minimum 4.5:1 ratio for normal text, 3:1 for large text and telemetry gauges).
* **Breakpoints:**
  - Desktop / Ultrawide (> 1200px): Full 4-card KPI strip and 2-column analytics grid.
  - Tablet (768px – 1199px): 2x2 KPI grid, stacked 1-column analytics.
  - Mobile (< 767px): Single-column scrollable stack with collapsible graph controls.
