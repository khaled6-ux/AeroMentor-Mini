
# AeroMentor Mini

A Python-based educational prototype for intelligent
flight performance analysis.

## Project Overview

AeroMentor Mini generates synthetic flight telemetry,
analyzes flight performance, and provides rule-based
feedback.

This project is a learning prototype supporting the
development of AeroMentor, a proposed intelligent
flight-training analysis system.

## Features

- Synthetic flight data generation
- Altitude, airspeed, heading, pitch, and roll measurements
- Mean Absolute Error (MAE) calculations
- Rule-based flight feedback
- Basic automated tests

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn (planned for Machine Learning)

## Getting Started

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate flight data:

```bash
python src/generate_flight_data.py
```

Analyze the generated flight:

```bash
python src/analyze_flight.py
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Roadmap

1. Synthetic flight analysis
2. Machine Learning classification
3. Explainable AI
4. FlightGear simulator integration

## Limitations

The current data is synthetic and does not represent
physically accurate aircraft behavior.

Performance tolerances are experimental and are not
certified aviation training standards.
