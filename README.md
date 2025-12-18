# Legislative Voting Behavior Prediction (Chile)

This project develops an end-to-end data pipeline to extract, process, and model open legislative voting data from the Chilean Congress, with the goal of predicting parliamentary voting behavior.

The repository was developed as a final degree project in Data Science Engineering and emphasizes reproducibility, interpretability, and structured data workflows.

---

## Problem Statement

Understanding and predicting legislative voting behavior is a relevant problem in political science and applied data science. Parliamentary votes reflect ideological alignment, party discipline, and institutional dynamics.

The goal of this project is to enhance transparency for citizens by developing models capable of predicting roll-call votes of Chilean parliamentarians. We argue that identifying the features that drive voting behavior can strengthen democratic accountability and citizen oversight of legislative processes.

The objectives of this project are to:
- Build a reproducible pipeline that collects and processes open voting data from the Chilean Congress.
- Extract and process plenary session transcripts to identify parliamentary stances.
- Engineer meaningful features at the legislator and bill level.
- Train and evaluate predictive models of voting behavior.

---

## Research Question

Which institutional, partisan, and individual features best predict parliamentary voting behavior?

---

## Data Sources

The data is obtained from open data published, including:

- Open voting records and legislative metadata  
- Parliamentary transparency and open data portals  
- Plenary session transcripts  
- Complementary institutional information from the Biblioteca del Congreso Nacional  

All data sources are publicly available and automatically processed by the pipeline.

---

## Pipeline Overview

The project follows a modular and extensible pipeline architecture:

### 1. Data Extraction
- Automated retrieval of roll-call votes and legislative metadata.
- Standardization of formats, identifiers, and time references.

### 2. Data Processing
- Cleaning and normalization of voting records.
- Extraction of speeches from plenary session transcripts.
- NLP processing to infer parliamentary stances.
- Bill topic modeling using BERTopic.
- Construction of legislator-level and vote-level datasets.
- Handling of missing values and structural inconsistencies.

### 3. Feature Engineering
- Party discipline metrics (including Rice Index).
- Historical voting behavior features.
- Contextual and temporal variables.

### 4. Modeling
- Supervised learning models to predict voting outcomes.
- Evaluation using appropriate classification metrics.
- Comparison between baseline and more complex models.
- Model training across different contexts (legislative periods and thematic domains).

### 5. Analysis
- Model interpretability analysis using SHAP values.
- Comparison of feature importance across scenarios.

---

## Methodology

- Exploratory Data Analysis (EDA) to identify voting patterns.
- Feature selection informed by political science theory.
- Cross-validation to ensure robust and stable evaluation.
- Preference for interpretable models over black-box performance.

---

## Results

The models achieve predictive performance above baseline levels, indicating that historical voting behavior and party affiliation provide strong explanatory signals for parliamentary voting decisions.

Detailed results, evaluation metrics, and visualizations are available in the notebooks included in this repository.

---

## Limitations

- External political context (media coverage, public opinion, electoral incentives) is not incorporated.
- NLP-based stance extraction is limited by the quality and structure of available transcripts.
- Results are specific to the Chilean legislative system and may not generalize directly to other countries.

---

## Future Work

- Incorporate external political and media context.
- Improve stance detection using more advanced NLP models.
- Deploy the pipeline as an automated and periodically updated system.
- Extend the framework to comparative legislative settings.

---

## Repository Structure
