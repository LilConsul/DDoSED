# Lightweight ML-Based DDoS Detection Dashboard Design

## 1. Overview

This project will be a lightweight, research-backed DDoS detection demo application built on the provided dataset. The system will classify traffic into three classes:

- normal
- SYN flood
- UDP flood

The project is intended to balance academic value and demo quality. It will compare several lightweight machine learning models, compare them against a simple rule-based baseline, and present the results in a small local dashboard.

The dashboard is a demonstration interface, not a production intrusion detection or mitigation system.

## 2. Goals

The project should answer the following questions:

- Can lightweight ML models detect DDoS attacks accurately?
- Which traffic features are the most useful?
- How does ML-based detection compare to rule-based methods?

The first version should also provide a clear visual demonstration suitable for a group project presentation.

## 3. Scope

### In scope

- loading and inspecting the provided dataset
- mapping labels into three target classes
- selecting a lightweight, behavior-focused feature set
- training and comparing multiple lightweight ML models
- implementing a simple rule-based baseline
- evaluating models with standard classification metrics
- presenting results in a local dashboard
- showing predictions on predefined static CSV samples
- simulating mitigation recommendations in the dashboard
- optionally replaying dataset windows as simulated traffic

### Out of scope

- live packet capture
- deployment on a real network
- real firewall integration
- real blocking or mitigation actions
- generation of real DDoS traffic
- arbitrary user file upload in the first version

## 4. Dataset and Labels

The dataset is already accessible through [`src/load_dataset.py`](src/load_dataset.py:6). Based on current inspection, the `Attack Type` column appears suitable as the primary target source.

Current class counts:

- `SYN Flood`: 16168
- `No Attack`: 16014
- `UDP Flood`: 16010

This is a well-balanced three-class setup.

### Target mapping

The target classes will be normalized as follows:

- `No Attack` → `normal`
- `SYN Flood` → `syn_flood`
- `UDP Flood` → `udp_flood`

If inconsistencies are later found between `Attack Type` and other label columns, `Attack Type` remains the authoritative source for the first version unless data inspection proves otherwise.

## 5. System Architecture

The system will be organized into the following logical parts:

### Data layer

Responsible for:

- loading the dataset
- validating schema
- cleaning rows and handling missing values
- normalizing labels
- splitting data into train, validation, and test sets

### Feature layer

Responsible for:

- selecting allowed features
- encoding categorical values where needed
- scaling numeric values where needed
- producing both a fuller safe feature set and a reduced lightweight feature set

### Modeling layer

Responsible for:

- training several lightweight classifiers
- storing model configurations and results
- selecting the best model for dashboard inference

### Baseline layer

Responsible for:

- implementing a simple rule-based detector
- providing a non-ML comparison point for the research questions

### Evaluation layer

Responsible for:

- computing metrics
- generating confusion matrices
- comparing models
- summarizing feature importance or coefficient-based rankings
- measuring inference efficiency where practical

### Dashboard layer

Responsible for:

- presenting model comparison results
- showing metrics and visual summaries
- displaying predictions on predefined static CSV samples
- optionally replaying dataset windows as simulated traffic
- showing mitigation recommendations

### Mitigation layer

Responsible for:

- mapping predicted class and confidence to simulated defensive recommendations
- presenting those recommendations in a safe, educational way

## 6. Feature Policy

The project should prioritize lightweight, explainable, behavior-based features rather than using every available column.

### Keep in the first version

Primary candidates include:

- `Protocol`
- `Source Port`
- `Destination Port`
- `Packet Size`
- `Payload Length`
- `Flow Duration`
- `Bytes in Flow`
- `Packets in Flow`
- `Average Packet Size`
- `Inter-Arrival Time`
- `Rate of Packets`
- `Unique Source Count`
- `Unique Destination Count`

These features are aligned with traffic behavior and are more defensible for DDoS detection than identity-based metadata.

### Drop in the first version

The following fields should be excluded initially:

- raw `Source IP`
- raw `Destination IP`
- `Timestamp`
- `Device Type`
- `Operating System`
- `Firmware Version`
- `Anomaly Score`
- any field that appears to leak labels directly
- any field that is only meaningful after the attack is already known

### Why raw IPs are excluded

Raw IP addresses can cause the model to memorize specific identities rather than learn attack behavior. That weakens generalization and makes the research claim less convincing. For this reason, the design keeps only aggregate IP-related features such as unique source and destination counts, while excluding raw addresses.

### Why `Anomaly Score` is excluded

`Anomaly Score` likely encodes prior detection logic or derived knowledge that would make the comparison unfair. Including it would weaken the validity of the research conclusions.

## 7. Lightweight Modeling Strategy

The project should compare several lightweight models rather than relying on a single classifier.

Candidate models:

- logistic regression
- decision tree
- random forest
- naive bayes
- k-nearest neighbors

The exact final set may be reduced during implementation if needed, but the design assumes multiple lightweight models will be compared.

### Model selection principle

The project should prefer simpler models when performance is close. The goal is not only high accuracy, but also:

- low complexity
- explainability
- lower overfitting risk
- suitability for a lightweight educational detector

### Feature-set comparison

To support the research direction, the project should compare:

- a fuller safe feature set
- a reduced lightweight feature set

This allows the team to evaluate whether fewer features can still provide strong performance.

## 8. Rule-Based Baseline

A simple rule-based baseline should be implemented for comparison with ML models.

The baseline may use heuristics based on:

- packet rate
- protocol
- packet size
- flow duration
- source/destination diversity

The baseline does not need to be sophisticated. Its purpose is to provide a clear comparison point for the research question about ML-based detection versus rule-based methods.

## 9. Data Flow

The expected processing flow is:

1. load dataset
2. inspect and normalize labels
3. remove excluded columns
4. prepare safe feature sets
5. split into train, validation, and test sets
6. preprocess features
7. train multiple ML models
8. evaluate ML models and rule baseline
9. select best model
10. save model artifacts and evaluation outputs
11. present results in the dashboard
12. run predictions on predefined static CSV samples
13. optionally replay dataset windows as simulated traffic

## 10. Dashboard Design

The dashboard should be local and lightweight. It should focus on clarity rather than production-grade complexity.

### Required dashboard views

- project overview
- dataset summary
- model comparison table
- confusion matrix and per-class metrics
- feature importance or coefficient ranking
- prediction results for predefined static CSV samples
- mitigation recommendation panel

### Optional dashboard view

- replay mode that presents dataset windows as simulated traffic over time

### Static sample prediction

Instead of arbitrary file upload, the first version should use predefined static CSV samples bundled with the project. This keeps the demo controlled, simpler to implement, and easier to present.

## 11. Mitigation Simulation

The project should not perform real mitigation. It should only simulate recommendations based on predictions.

Examples:

- predicted `syn_flood` with high confidence → recommend SYN-focused filtering or rate limiting
- predicted `udp_flood` with high confidence → recommend UDP-focused filtering or rate limiting
- low-confidence prediction → recommend analyst review
- predicted `normal` → recommend no action

These recommendations are educational outputs only.

## 12. Evaluation Plan

The project should report:

- accuracy
- macro precision
- macro recall
- macro F1
- confusion matrix
- per-class metrics
- model comparison summary
- inference speed or relative efficiency where practical

The evaluation should compare:

- multiple lightweight ML models
- rule-based baseline
- fuller safe feature set versus reduced lightweight feature set

## 13. Success Criteria

The first version is successful if it:

- correctly performs three-class classification
- compares multiple lightweight ML models
- compares ML against a rule-based baseline
- uses a reduced, behavior-focused feature strategy
- presents results in a local dashboard
- demonstrates predictions on predefined static CSV samples
- provides simulated mitigation recommendations
- supports the project’s research questions with measurable results

## 14. Risks and Constraints

### Risks

- some dataset columns may contain hidden leakage
- some features may inflate performance without improving generalization
- too many features may conflict with the lightweight goal
- dashboard work could consume time better spent on model quality

### Constraints

- the project should remain educational and safe
- the first version should avoid live traffic capture and real attack generation
- the implementation should stay manageable for a three-person student team

## 15. Recommended Implementation Direction

The recommended implementation direction is:

- build an offline training and evaluation pipeline first
- compare several lightweight models
- establish the rule-based baseline
- identify the best lightweight model and reduced feature set
- build a local dashboard around saved results and predefined sample predictions
- add replay mode only if time allows

This keeps the project academically strong, technically manageable, and presentation-friendly.