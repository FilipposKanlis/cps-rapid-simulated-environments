# cps-rapid-simulated-environments
Python scripts and simulation assets for a rapid-development methodology of simulated environments to detect unexpected behaviors in Cyber-Physical Systems (CPS).

---

# Abstract
This thesis focuses on the development of a methodology for the rapid creation and utilization of simulated Cyber-Physical Systems (CPS) using low-code and model-driven tools. The main objective is to accelerate the design, adaptation, and testing of system scenarios without the need for physical hardware.

The proposed methodology enables the construction of fully functional virtual environments through the **Locsys** platform, which integrates environment design tools (EnvMaker, EnvPop, AppCreator) with communication and data management infrastructures (**MQTT**, **Redis**, **MongoDB**). These environments provide a flexible framework for simulating and evaluating sensor-based algorithms in a controlled and reproducible way.

As a demonstration of the methodology, a **rule-based behavior detection system** was implemented to monitor movement patterns and detect deviations from normal operation across three virtual environments: a **Smart Home**, a **Farm**, and a **City**.

The results confirm that the proposed approach supports rapid development, experimentation, and evaluation of CPS, offering a reliable foundation for studying and validating human-centered behaviors in simulated and privacy-preserving environments.

---

# Project Overview

This repository contains:

- Python rule-based behavior recognition scripts  
- Pattern-matching and sensor event processing  
- MongoDB logging utilities  
- Rule-based similarity scoring  
- Experiments in Smart Home, Smart Farm, and Smart City environments  

The virtual environments were created using the **Locsys CPS Simulation Platform** (private workspace):  
🔗 https://locsys.issel.ee.auth.gr/projects/690dfe75d807b62d0b187c53 *(access requires credentials)*

---

# Locsys Tools Used
- **EnvMaker** – Environment grid/layout creation  
- **EnvPop** – Sensor and actor placement  
- **AppCreator** – Flow-based behavior programming (robot paths, triggers, logic)  
- **AppDeployer** – Execution via Docker containers (Redis + MQTT + goal executor)

---

# Installation

```bash
pip install commlib redis pymongo matplotlib numpy
```

---

# Set Environment Variables

Before running any script (home, farm, or city), set:

```bash
# Mode options: collect | detect
export MODE=collect
# or
export MODE=detect

# Pattern tag (used in COLLECT mode to store pattern)
export PATTERN=pattern1

# MongoDB connection (required for both modes)
export MONGO_URI="your_mongodb_uri_here"
```

---

# Run Scripts (Home, Farm, City)

Each file corresponds to one simulated environment:

- `home_rulebased_final.py`
- `farm_rulebased_final.py`
- `city_rulebased_final.py`

Run any of them directly:

```bash
python home_rulebased_final.py
python farm_rulebased_final.py
python city_rulebased_final.py
```

---

# Usage Examples

### 🟦 **Collect Mode (record a new pattern)**  
Used to capture a sequence of movements and store it in MongoDB.

```bash
export MODE=collect
export PATTERN=pattern3
python farm_rulebased_final.py
```

### 🟩 **Detect Mode (recognize behavior)**  
Compares real-time sensor events with stored patterns.

```bash
export MODE=detect
python farm_rulebased_final.py
```

Alternatively (inline variables):

```bash
PATTERN=pattern_3 MODE=collect python farm_rulebased_final.py
MODE=detect python farm_rulebased_final.py
```

---

# Repository Structure

```
├── home_rulebased_final.py
├── farm_rulebased_final.py
├── city_rulebased_final.py
├── README.md
└── .gitignore
```

---

# Notes
- Locsys environments (Smart Home, Farm, City) run through private deployment spaces.  
- The scripts here connect to **Redis** and **MongoDB Atlas** for event streaming and logging.  
- Designed for reproducibility, modularity, and privacy-preserving behavior analytics.

---

# License
No license included — default copyright applies.
