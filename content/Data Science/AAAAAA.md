---
title: Personlige læringsmål
publish: false
draft: true
---



### Korrelation: Hvorfor Pearson fejler

En klassisk fejl ville være at bruge **Pearsons Korrelation** til at måle sammenhængen mellem lysstyrke og præcision. Pearson forudsætter lineære sammenhænge og normalfordeling.

- **Realiteten:** Sammenhængen mellem lys og OCR-kvalitet er ikke lineær. Det går godt længe, og så dykker det pludseligt ("Death Zone").
    
- **Løsning (Binning & Spearman):** I stedet for at lede efter en ret linje, har jeg inddelt data i "Buckets" (Low, Mid, High Light) og analyseret dem som ordinale data. Dette svarer metodisk til en **Spearman Rank Correlation**, hvor vi kigger på rangordning (bliver det værre, når det bliver mørkere?) frem for lineære værdier.
    

### Outlier Detection: Multivariate Anomalier

I min "Cleaning"-sektion nævnte jeg simple outliers. Men de farligste fejl er **Multivariate Outliers** – datapunkter, der ser normale ud hver for sig, men er umulige sammen.

- _Eksempel:_ En scanning med `confidence = 0.99` (meget høj) men `levenshtein = 5` (helt forkert tekst).
    
- **Betydning:** Dette er ikke bare en fejl; det er en "Hallucination". Modellen er skråsikker på noget, der er forkert. Disse "Confident Failures" er de vigtigste datapunkter at hive ud til et _retraining set_, da de straffer brugeroplevelsen hårdest.





### Phase 1: The Data Layer (Files: `01_corrections_schema.md`, `generate_data.py`)

- **Theory Applied:** _Passive Collection_, _Tidy Data_, _Labeling (Weak vs Strong)_.
    
- **The Action:**
    
    - Define `CorrectionEvent` (Schema) $\rightarrow$ Proof of **Passive Data Collection**.
        
    - Structure the CSV columns $\rightarrow$ Proof of **Tidy Data Principles** (1 row = 1 observation).
        
    - Capture `plate_pred` vs `plate_true` $\rightarrow$ Proof of **Supervised Learning Labels** (Weak vs Strong).
        

### Phase 2: The "Cleaning" Simulation (File: `dashboard.py` - Part 1)

- **Theory Applied:** _Data Preparation_, _Handling Missing Values_, _Deduplication_.
    
- **The Action:**
    
    - Load CSV with `pandas`.
        
    - Code a visibly explicit "Cleanup Block" (even if data is clean) $\rightarrow$ Proof of **Data Prep (80% rule)**.
        
    - Convert timestamps and category types $\rightarrow$ Proof of **Transforming**.
        

### Phase 3: The Analytics Dashboard (File: `dashboard.py` - Part 2)

- **Theory Applied:** _EDA_, _Summary Stats_, _Correlations_, _Outlier Analysis_.
    
- **The Action:**
    
    - **Chart 1:** Confidence vs. Correctness Scatter $\rightarrow$ Proof of **Outlier Analysis** (identifying the "garbage").
        
    - **Chart 2:** Accuracy by Lighting (Bar) $\rightarrow$ Proof of **Bivariate Analysis** (Categorical vs Numeric).
        
    - **Chart 3:** Confusion Matrix (Table) $\rightarrow$ Proof of **Chi-Square/Heatmap** logic for categorical errors.
        

### Phase 4: The Validation Artifacts (Files: `04_metrics.md`, `05_loop_diagram.png`)

- **Theory Applied:** _Validation_, _Evaluation Metrics_, _Human-in-the-Loop_.
    
- **The Action:**
    
    - Calculate F1 & Levenshtein $\rightarrow$ Proof of **Metric Selection** (Accuracy is for colors, Edit Distance is for text).
        
    - Draw the Feedback Loop $\rightarrow$ Proof of **Continuous Supervised Learning**.