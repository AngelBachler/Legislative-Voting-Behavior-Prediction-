# Legislative Voting Behavior Prediction (Chile)

This project implements a full end-to-end data science pipeline to predict legislative voting behavior using real-world parliamentary data.

The work combines:
- Designing and implementing a reproducible data pipeline from raw open data
- Cleaning, normalizing, and integrating heterogeneous legislative datasets
- Engineering predictive features at legislator, bill, and vote levels
- Applying NLP techniques to extract parliamentary stances from plenary transcripts
- Fine-tuning a Chilean Spanish BERT model for domain-specific stance classification
- Training and evaluating supervised classification models (including XGBoost and CatBoost)
- Interpreting model outputs using SHAP values
- Communicating results through visualizations and an academic presentation in English

The project demonstrates applied skills in data engineering, machine learning, NLP, and analytical reasoning using real institutional data.

---

## Tech Stack

- Programming: Python
- Data Processing: Pandas, NumPy
- Machine Learning: Scikit-learn, XGBoost, CatBoost
- NLP: Hugging Face Transformers, BERTopic, BERT fine-tuning, LLMs
- Model Interpretability: SHAP
- Data Visualization: Matplotlib, Seaborn
- Version Control: Git

## Academic context
This project was developed as part of my undergraduate degree in Data Science Engineering at Pontificia Universidad Católica de Chile 
and concluded with an academic presentation delivered in English.

---

## Problem Statement

Understanding and predicting legislative voting behavior is a relevant problem in political science and applied data science. Parliamentary votes reflect ideological alignment, party discipline, and institutional dynamics.

The goal of this project is to enhance transparency for citizens by developing models capable of predicting roll-call votes of Chilean parliamentarians. Identifying the features that drive voting behavior can help improve democratic transparency 
by enabling citizen oversight of legislative decision-making.

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

The data is obtained from publicly available open data sources, including:

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
## Trained Models

To extract parliamentary stances from plenary session transcripts, we initially used the Spanish BERT model **dccuchile/bert-base-spanish-wwm-uncased**, available on Hugging Face.

Although this model provides strong performance on general Spanish-language tasks, it was pre-trained primarily on informal textual sources (e.g., tweets). As a result, it struggled to correctly infer negative or oppositional stances expressed in the formal and diplomatic language used in parliamentary debates.

To address this limitation, we fine-tuned the model on a domain-specific corpus derived from parliamentary dialogue annotations. This fine-tuning significantly improved stance classification performance, particularly for negative and neutral classes, as shown in the figures below.

<img width="748" height="440" alt="image" src="https://github.com/user-attachments/assets/2c960bf5-0a43-41bf-88d7-8b3b752ccabd" />

<img width="678" height="489" alt="image" src="https://github.com/user-attachments/assets/add9efa7-d49c-4231-bc55-8624fccadb54" />

Due to size and reproducibility considerations, fine-tuned model artifacts are not included in the repository. All results can be reproduced using the training pipeline provided.

--- 

## Key Outcomes

- Developed a scalable and reproducible pipeline for legislative data analysis
- Achieved stable predictive performance above baseline across multiple legislative periods
- Identified domain-dependent drivers of voting behavior using SHAP interpretability

---
## Results

The models achieve predictive performance above baseline levels, indicating that historical voting behavior and party affiliation provide strong explanatory signals for parliamentary voting decisions.

Overall, the models demonstrate strong and stable performance in predicting roll-call votes across different legislative periods, as shown in the following figure.

<img width="673" height="417" alt="image" src="https://github.com/user-attachments/assets/304b8b87-8f8a-4ed9-8328-e387b0f58642" />

The second figure presents the performance of an XGBoost model across different bill topics (Security, Pensions, Education, and Health).

<img width="637" height="451" alt="image" src="https://github.com/user-attachments/assets/f0045c56-3d4f-408f-9193-4a1f591fa637" />

Interpretability analysis using SHAP values reveals that the relevance of features varies across policy domains. For instance, stance-related features play a more significant role in topics such as Pensions and Security than in areas like Health.

<img width="957" height="520" alt="image" src="https://github.com/user-attachments/assets/1a042751-3b91-4b58-a496-9b34ba8634e7" />

<img width="438" height="531" alt="image" src="https://github.com/user-attachments/assets/07d39b26-b6e3-440f-92e7-b02fc5a8776c" />

This suggests that issue salience and public opinion may differentially shape voting behavior depending on the policy domain, an aspect that warrants further investigation.

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



