# 🚗 Shift-by-Wire Fault Detection using Explainable AI (SHAP & LIME)

> **Explainable AI–based fault detection for Shift-by-Wire automotive systems using Isolation Forest, SHAP, and LIME.**

---

## 📘 Overview
This project demonstrates **data-driven fault detection** and **explainability** for a simulated **Shift-by-Wire (SBW)** system — an advanced automotive mechanism that replaces mechanical linkages with electronic controls.

The pipeline uses **unsupervised machine learning (Isolation Forest)** to detect anomalies in SBW sensor behavior and employs **SHAP** and **LIME** to interpret and visualize *why* faults are detected — bridging the gap between **AI predictions** and **engineering understanding**.

---

## 🧭 Objectives
- Simulate realistic **Shift-by-Wire sensor data** (torque, actuator, clutch pressure, gear).  
- Engineer **dynamic features** that capture system health and transitions.  
- Detect **faults** using **Isolation Forest** (unsupervised anomaly detection).  
- Explain **model reasoning** using **SHAP** (global & local) and **LIME** (local).  
- Visualize fault behavior with **interpretable plots** for engineers.

---

## ⚙️ System Architecture

+---------------------------------------------------+
|          Synthetic Shift-by-Wire Dataset          |
|  torque_demand, torque_actual, actuator_pos, etc. |
+-----------------------------+---------------------+
                              |
                              v
                  Feature Engineering (KPIs)
     ------------------------------------------------
     |  torque_error_rm, torque_rate_rm, pressure_rate, actuator_pos  |
     ------------------------------------------------
                              |
                              v
                 Unsupervised Fault Detection
                    [ Isolation Forest Model ]
                              |
                              v
                   Explainability Components
                     [ SHAP + LIME Analysis ]
                              |
                              v
                     Visualization & Insights
        (Torque-Time plots, SHAP summary, LIME table, etc.)


---

## 🧮 Dataset Description

| Feature | Description | Unit |
|----------|--------------|------|
| `time_s` | Timestamp | s |
| `gear` | Gear position (1–6) | – |
| `torque_demand` | Driver-requested torque | Nm |
| `torque_actual` | Delivered torque (with injected faults) | Nm |
| `actuator_pos` | Actuator position | % |
| `clutch_pressure` | Hydraulic clutch pressure | bar |

During generation, random torque dropouts are introduced to simulate actuator lags and signal inconsistencies.

---

## 🧠 Feature Engineering

| Feature | Formula | Purpose |
|----------|----------|----------|
| `torque_error_rm` | Rolling mean of (demand − actual) | Detect actuator or control lag |
| `torque_rate_rm` | Rolling mean of Δtorque/Δtime | Capture response speed |
| `pressure_rate` | Δpressure/Δtime | Identify hydraulic instability |
| `actuator_pos` | Raw actuator position | Monitor displacement behavior |

These features represent the **core KPIs** that define SBW health.

---

## 🤖 Fault Detection Model

| Component | Description |
|------------|-------------|
| **Algorithm** | Isolation Forest |
| **Type** | Unsupervised anomaly detection |
| **Contamination** | 0.05 (≈5% expected faults) |
| **Goal** | Learn normal sensor behavior and flag deviations as faults |

```python
from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.05, random_state=42)
df['fault_flag'] = model.fit_predict(X_scaled)
```

📊 Visualization

| Visualization    | Insight                                                      |
| ---------------- | ------------------------------------------------------------ |
| Torque-Time Plot | Fault points (red) show torque dropouts                      |
| SHAP Summary     | `pressure_rate` & `torque_error_rm` dominate fault detection |
| SHAP Waterfall   | Breaks down one prediction’s sensor-level reasoning          |
| LIME Table       | Human-readable explanation of a local fault event            |

📈 Key Results
| Metric                    | Value / Insight                           |
| ------------------------- | ----------------------------------------- |
| Dataset Size              | 3000 samples                              |
| Features                  | 4 engineered KPIs                         |
| Detected Faults           | ≈5%                                       |
| Most Influential Features | `pressure_rate`, `torque_error_rm`        |
| Explainability Outcome    | Transparent, sensor-level fault reasoning |

🧰 Tech Stack

| Category       | Tools               |
| -------------- | ------------------- |
| Language       | Python 3            |
| ML Framework   | Scikit-learn        |
| Explainability | SHAP, LIME          |
| Visualization  | Matplotlib, Seaborn |
| Data Handling  | Pandas, NumPy       |
| Environment    | Jupyter Notebook    |

The model detected anomalies primarily when actuator position and torque rate deviated from normal operation.
SHAP confirmed these features’ dominant impact, while LIME provided localized explanations showing clutch pressure spikes and torque errors as the root cause of specific faults.
