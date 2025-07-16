# AI Data Science Pipeline – Agent Specifications

## OrchestrationAgent

### Role
Central controller that manages pipeline state, routing logic, rollback, agent execution, and session coordination.

### Input
- `user_metadata`: file path, selected target, user preferences
- `session_state`: current phase, approved outputs

### Output
- Triggers next agent
- Logs rollback snapshots and phase state

### Core Logic
- Maintains `phase_queue`
- Invokes each agent's `run(context)`
- Stores snapshot before/after each agent run
- Checks Streamlit approval before proceeding
- Implements rollback: clears downstream context and resets pipeline state

### Memory
- Loads past user preferences
- Updates project timeline, phase timestamps

### UI
- Displays current phase, logs, rollback controls

### Failure Handling
- If agent fails: logs, alerts UI, halts pipeline

---

## DataIngestionAgent

### Role:
Validates and ingests structured data files (`.csv`, `.xlsx`), parses schema, extracts target column.

### Inputs
- `uploaded_file`
- `target_variable`

### Outputs
- `context.data`: `pandas.DataFrame`
- `context.metadata`: `{ columns, dtypes, n_rows, null_map }`

### Core Logic
- Read via `pandas.read_csv()` or `read_excel()`
- Validates:
    - Column presence
    - Target not all nulls
    - Unique IDs (optional)
- Logs file metadata

### Memory
- Stores file hash, column names, shape
- Fetches similar past datasets

### UI
- Shows table preview, column detection summary
- User approves target column or reselects

---

## DataProfilingAgent

### Role
Generates statistical summary, data types, distributions, and feature risk alerts.

### Inputs
- `context.data`

### Outputs
- `context.profile_summary`
- `context.llm_insights` (narrative)

### Core Logic
- Uses `pandas-profiling` or `ydata-profiling`
- Extracts:
    - Missing %
    - Unique values
    - Feature types
    - Outliers (via IQR)

### Memory
- Stores column stats
- Retrieves past alerts for similar schemas

### UI
- Auto-suggest: columns to drop, risk flags
- Generate natural-language summary

---

## DataCleaningAgent

### Role
Identifies and applies cleaning operations with code-based + LLM-based reasoning.

### Inputs
- `profile_summary`
- `user_feedback` (optional)

### Output
- `cleaned_data`
- Cleaning code snippet (logged)

### Core Logic
- Suggests:
    - Imputation
    - Type Fixes
    - Outlier capping
    - Null % thresholds
- LLM explains rationale
- Accepts custom cleaning code from user

### Memory
- Stores action→outcome logs (e.g., "dropping zip reduced MAE by 5%")
- Suggests prior rules for similar columns

### UI
- Preview data diff
- Approve, override, or upload cleaning script

---

## FeatureSelectionAgent

### Role
Reduces feature space by removing low-signal, redundant, or collinear features.

### Inputs
- `cleaned_data`
- `target`

### Outputs
- `selected_features`
- Feature mask/log

### Core Logic
- Rule-based:
    - Variance threshold
    - Correlation matrix
    - Null % > 90
- Optional LLM: "Which features might cause leakage?"

### Memory
- List included/excluded features
- Click to reinstate dropped ones

---

## FeatureEngineeringAgent

### Role
Suggests and applies new features using statistical logic + LLM insights.

### Input
- `selected_features`
- Optional user suggestions

### Output
- `context.data_enriched`

### Core Logic
- Adds:
    - Ratio features
    - Binned versions
    - Interaction terms
    - Lags/rolling stats (for time series)
- LLM suggests domain-specific transforms
- User can upload feature code

### Memory
- Fetches previously successful features

### UI
- Show before/after comparison
- Accept/reject individual features

---

## ModelingAgent

### Role
Trains supervised models, selects best via performance + suitability.

### Inputs
- `data-enriched`
- `target`

### Outputs
- `model.pkl`
- Training logs

### Core Logic
- Task type detection (classification/regression)
- Runs 3–5 models:
    - Logistic, XGBoost, RF, etc.
- Light tuning (GridSearchCV, randomized)
- Uses Modeling Policy

### Memory
- Compares against past model performance
- Stores model scores, params

### UI
- Shows model leaderboard
- User can select preferred model

---

## TimeSeriesAgent

### Role
Handles end-to-end time-series forecasting with multivariate support.

### Input
- Data with datetime + target

### Output
- Forecast table (next 90d)
- Metrics, plots

### Core Logic
- Resampling, fill missing
- Feature gen (lags, datetime parts)
- Models:
    - Prophet, ARIMA, XGBoost
- Walk-forward CV

### Memory
- logs seasonality patterns
- Retrieves success history on similar series

### UI
- Shows forecast curve + CI
- User adjusts horizon or seasonality toggle

---

## EvaluationAgent

### Role
Evaluates model performance, generates insights + diagnostics.

### Inputs
- Trained Model
- Holdout Data

### Outputs
- Metric Summary
- Diagnostic plots
- Interpretation narrative

### Core Logic
- Metrics: AUC, F1, R², MAE
- SHAP summary plot
- Confusion matrix, residual plot
- LLM explains what metrics mean

### Memory
- Logs metric thresholds
- Stores feedback on past model interpretability

### UI
Users prioritize metrics (e.g., "optimize for recall")

---

## ReportGenAgent

### Role
Generates an md report describing pipeline steps, model results, insights, and recommendations.

### Inputs
- Logs, metrics, model summary, user feedback

### Outputs
- `report.md`

### Core Logic
- Sectioned structure:
    - Problem summary
    - Data summary
    - Model results
    - SHAP + visualizations
    - Recommendations

- LLM generates natural-language interpretations

### Memory
- Recalls successful past explanations
- Adjusts tone based on user profile

### UI
- Preview report + download

---

## MemoryAgent

### Role
Stores and retrieves structured logs + embeddings of user, data, and modeling behavior.

### Inputs
Events from all agents

### Outputs
- query_memory(query_str)
- add_memory(content, tags)

### Core Logic
- Vector DB (e.g., ChromaDB)
- Symbolic JSONL/Parquet for structured data
- Supports semantic + exact retrieval

### Storage Schema
- Dataset profile
- Feature success/failure
- Model performance
- User preferences, overrides

### UI
- N/A (Internal service)















