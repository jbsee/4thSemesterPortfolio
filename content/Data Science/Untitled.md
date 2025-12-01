---
title: 6. Perspekhgtivering og refleksion
publish: false
draft: true
---




## Labeling Data for Supervised Learning

### Core Idea

- In supervised learning, **labels = the outcome we want to predict or classify**.
- Labels orient the model — they tell it _what correct looks like_.

|                                           |                                                                                           |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- |
| Embedded tasks (e.g. CAPTCHA corrections) | Users unknowingly label data during regular interactions. Cheap, real-world, large-scale. |
Your app is an embedded labeling system:  
User confirms or corrects plate text or color → stored as labeled ground truth.



**Limited data** on our current test: Missing real-world variation (rare formats, special plate types).

Solution: track context metadata (device, lighting, plate format, country code) to see where your dataset lacks coverage.

Each CorrectionEvent should log:

- `plate_predicted` vs `plate_corrected` (label)
- `color_predicted` vs `color_corrected`
- Context: `device`, `lighting_level`, `confidence`, `image_hash`, `country_code_guess`
- Timestamp, model version

## Passive Collection of Training Data

### Core Idea

Passive data collection = gathering training data automatically as the system is being used, without manual labeling effort.

For your app:  
Each time a user corrects a misread plate or color, they are unknowingly generating labeled, context-rich training data. You don’t prompt them for annotation; it's part of normal use.

### Benefits of Passive Collection

- **Scale:** System can accumulate large volumes of labeled data over time, without annotation cost.
- **Real-world authenticity:** Data reflects real conditions: lighting, angle, device, weather, plate styles, regional formats.
- **Continuous updates:** Data arrives as user behavior changes, allowing for adaptation to new plate formats or distributions.

This is **passive supervised training data collection** in production.
### Challenges and Considerations

|Challenge|Relevance to your system|
|---|---|
|Representation|Must ensure data covers different plate styles, fonts, countries, lighting, colors, device types.|
|Shared meaning|Define what counts as correct color or correct plate format (avoid ambiguity in corrections).|
|Limit cases|Rare situations: reflective plates, angled shots, custom plate fonts, low-resolution images. Must capture them distinctly, not force-fit into generic classes.|

Hvis vores brugere kun tager billeder på en bestemt måde, blive vores model også kun trænet til den slags billeder. Risiko for overfitting


## Validating Models

### Core principle

A model must be tested on **data it was not trained on**, to see how well it performs on new, unseen cases.  
You validate to measure **generalization**, not memorization.

### Main validation structures

#### 1. Train/Test Split (Holdout Validation)

- Split dataset into two parts:
    - **Training data** → used to build the model
    - **Holdout (test) data** → never used during training, only used to evaluate performance
- Gives the most realistic measure of performance on future data.

Det kunne sagtens implementeres i denne løbende dataindsamling, en smule af dataen lægges til side for validering.

### What matters for your pipeline

Even if you're not training full models yet, **you can still validate your correction logic**:

- Measure accuracy on **unseen correction events** (not the same events used to tweak rules).
- If rules are optimized using first 100 corrections, test them against correction 101–150 without modifying logic.
- If you later fine-tune an OCR model using your collected data, split it into **train** and **test**; don't let corrections used for parameter tweaking count as proof of improvement.

|Term|Meaning|
|---|---|
|Training set|Data used to build/tune model or rule|
|Validation set|Used during tuning (not final proof)|
|Test/Holdout set|Never touched during training; used only to measure generalization|

### Exam-friendly sentence

Validation proves whether your system works **on data it didn’t learn from**.  
It is the guardrail against overfitting and blind optimism.


## Supervised, Unsupervised, and Reinforcement Learning

### Three learning paradigms

#### 1. Supervised Learning

- Goal: Predict a known outcome or classify into predefined labels.
- Requires labeled training data.
- Examples: regression, classification, deep learning.
- Your system: plate_text and plate_color detection + user corrections = supervised ground truth.

#### 2. Unsupervised Learning

- Goal: Find structure or patterns in data without predefined labels.
- Examples: clustering, dimensionality reduction, anomaly detection.
- Relevant to your use-case:
    - Clustering correction events by lighting/device/character-confusion (to find hidden patterns)
    - Detecting unusual failure types or edge cases (rare plate formats, reflective surfaces)

#### 3. Reinforcement Learning

- Goal: Learn behavior through feedback signals (rewards/punishment), not predefined labels.
- No direct use in your pipeline. Mention only for completeness if asked.

### Use in your pipeline

|Learning Type|Relevance|
|---|---|
|Supervised|Core of your correction-based learning system|
|Unsupervised|Useful later for analyzing error clusters, spotting edge conditions|
|Reinforcement|Not relevant for this project|

Your app is primarily **supervised**, with mild **unsupervised** potential for discovery of error patterns.



---
---

