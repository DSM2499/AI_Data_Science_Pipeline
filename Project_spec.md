# AI Data Science Pipeline – Comprehensive Specification
**System Goal**: Automate the full data science lifecycle using intelligent agents, with human-in-the-loop control, long-term memory, and learning capabilities.

## System Architecture Overview
┌────────────────────────┐
│      Streamlit UI      │
└────────────┬───────────┘
             ▼
┌────────────────────────┐
│  UserInput & Metadata  │
│ - Target Variable      │
│ - Upload File Path     │
│ - User Preferences     │
└────────────┬───────────┘
             ▼
┌────────────────────────┐
│  Orchestration Agent   │  ← master control (aka TaskManager)
│ - Determines sequence  │
│ - Manages state        │
│ - Calls agents via .run│
│ - Receives results     │
└────────┬────┬──────────┘
         ▼    ▼
    ┌────────────┐    ┌────────────┐
    │ MemoryAgent│    │ FeedbackUI │ ← shows step-wise decisions + gets user approval
    └────┬───────┘    └────────────┘
         ▼
────────────────────────────────────────────
 PHASED EXECUTION (run by OrchestrationAgent)
────────────────────────────────────────────
         ▼
┌────────────────────┐
│ DataIngestionAgent │
└────────┬───────────┘
         ▼
┌────────────────────┐
│ DataProfilingAgent │ ← talks to MemoryAgent to log dataset schema
└────────┬───────────┘
         ▼
┌────────────────────┐
│ DataCleaningAgent  │ ← can call ProfilingAgent again for recheck
└────────┬───────────┘
         ▼
┌────────────────────┐
│ FeatureEngAgent    │ ← consults CleaningAgent for column metadata
└────────┬───────────┘
         ▼
┌────────────────────┐
│ ModelingAgent      │ ← reads user’s target, task type
└────────┬───────────┘
         ▼
┌────────────────────┐
│ EvaluationAgent    │ ← calculates metrics, plots, SHAP
└────────┬───────────┘
         ▼
┌────────────────────┐
│ ReportGenAgent     │ ← fetches Memory logs to generate insights
└────────┬───────────┘
         ▼
┌────────────────────┐
│ Final Output (md,  │
│  model.pkl, code)  │
└────────────────────┘


🔁 Feedback Loop:
→ After each phase, human can inspect results, approve, or suggest re-runs
→ Long-term memory persists past decisions, dataset patterns, model performance, and user preferences

📦 Artifacts:
→ Cleaned data, model object, full diagnostic report (PDF/HTML), reusable code scripts

## Pipeline Phases and Agents:
1. Data Ingestion:
    - Reads uploaded `.csv` or `.xlsx`
    - Validates schema, format, missing target column
    - Stores raw data snapshot
2. Data Profiling Agent:
    - Uses `pandas-profiling` or `ydata-profiling`
    - Identifies distributions, cardinality, missingness, feature types
    - Summarizes complexity and flags risks (leakage, imbalance)
3. Data Cleaning Agent:
    - Suggests and optionally applies:
        - Null imputation
        - Type Correction
        - Encodding (if needed)
    - Asks user to approve before mutation
4. Feature Engineering Agent:
    - Suggets:
        - Numerical transforms (log, binning, scaling)
        - Date decomposition
        - Interaction terms
    - Applies once user approves
5. Modeling Agent:
    - Determines whether the task is:
        - Regression
        - Classification
        - Time Series
    - Trains multiple models (AutoML style or pre-defined list)
    - Logs performance, hyperparameters, feature importances
6. Evaluation Agent:
    - Computes metrics (R², RMSE, AUC, etc.)
    - Generates plots (residuals, confusion matrix, etc.)
    - Scores interpretability (e.g., SHAP summary)
7. Reporting Agent
    - Compiles markdown with:
        - All steps taken
        - Key decisions and parameters
        - Final model summary
        - Insights + business implications
        - Recommendations
8. Memory Agent (Support):
    - Stores:
        - User preferences (e.g., modeling choices, columns removed)
        - Dataset History
        - Feature performance across datasets
    - Retrieves relevant info for all agents

## Interaction flow per phase
At the end of every phase:
    - Agent generates a summary report + options
    - UI prompts user:
        - Proceed
        - Re-run
        - Customize (e.g., column to drop, metric to optimize)
    - If user skips, agent proceeds with defaults.

## Failure Recovery + Data Insufficency:
If data is insufficient (e.g., low row count, poor signal):
- Agent triggers `Data Sufficiency Checker`
    - Suggests: more samples, higher variance columns, better labels
    - Provides example feature recommendations
    - Can ask user to upload additional data

## Memory and Learning
- Use `ChromaDB` for memory persistence.
- All agents log:
    - Dataset metadata
    - Feature performance
    - User override patterns
    - Common cleaning rules
- Agents can retrieve this to adapt behavior over time.
- Training loop: log model performance, allow human labeling on poor results, retrain incrementally.

## Deployment and Integration Rules:
- Full Python Backend
- Streamlit front-end with:
    - Phase checkpoint cards
    - Result visualizers (plotly/matplotlib)
    - User controls (accept/override/reset)
- Modular design: each agent callable as an independent service
- All outputs written to `project_output/` folder:
    - `cleaned_data.csv`, `model.pkl`, `report.md`, `scripts/`

## LLM Access by Agents
- Orchestration Agent
    - No LLM required
    - Reason: This agent is responsible for pipeline routing and phase sequencing — best done via deterministic logic.
    - Implementation: Python-based finite-state controller or rule-based task router.

- Data Ingestion Agent
    - No LLM Required
    Reason: Reading `.csv`, validating column presence, and parsing metadata is best done using pandas, no LLM needed.

- Data Profiling Agent
    - LLM Optional
    - Reason: Profiling is data-driven (`pandas-profiling`, `ydata-profiling`), but;
        - LLM can enhance this by summarizing profile findings in natural language for user inspection.
    - Recommendation:
        - Core profiling: rule-based
        - Summary + Insight Narrative: LLM

- Data Cleaning Agent
    - LLM required
    - Reason:
        - LLM can suggest context-aware cleaning strategies
        e.g. “This column may be dropped due to 99% nulls and low correlation”
        - Detect suspicious encodings or data leakage
    - LLM Roles:
        - Explain recommended actions to the user
        - Generate Python code snippets for custom cleaning steps (validated by user)

- Feature Engineering Agent
    - LLM required
    - Reason:
        - Feature generation can be aided by LLMs (examples, not limited to this):
            - “Create ratio of income to debt”
            - “Encode categorical variable with high cardinality using hashing”
        - LLM can also generate human-readable rationales for each feature.
    - Implementation:
        - Use LLM to suggest engineered features
        - Use code templates to apply them

- Modeling Agent
    - LLM Required
    - Reason:
        - Model training itself is rule-based (`sklearn`, `xgboost`, etc.)
        - LLM can assist in:
            - Choosing models based on task
            - Explaining differences between models to the user
            - Selecting hyperparameter grids (LLM + memory)

- Evalution Agent
    - LLM Required
    - Reason:
        - Metrics calculation is deterministic
        - But explaining results ("AUC is high but recall is low, indicating...") is ideal for LLMs.
        - Summarizing model errors and performance drivers can be automated with GPT
    - LLM Use:
        - Auto-generate analysis narratives
        - Interpret SHAP values
        - Suggest thresholds

- Reporting Agent
    - LLM Required
    - Reason:
        - LLMs can take structured logs, metrics, and outputs from other agents and generate:
            - Plain English summaries
            - Actionable insights
            - Recommendations
        - dynamically adjust tone based on audience (analyst vs exec)

- Memory Agent
    - LLM No Required
    - Optional: If you want memory to support semantic search over logs or feedback, you may encode chunks using LLM embeddings.

## Memory Subsystem

### Memory Backend
- Memory Type: Hybrid memory system:
    - Vector Store for similarity-based semantic recall (ChromaDB or FAISS)
    - Symbolic Store for structured metadata (SQLite or flat JSON/Parquet)

- Persistence: Stored in `memory/` folder with subfolders for:
    - `vector/` – embeddings + metadata
    - `symbolic/` – structured logs, config objects, past runs

### Memory Schema Design

#### Vector Store (Semantic Memory)
| Field          | Type          | Description                                                                |
| -------------- | ------------- | -------------------------------------------------------------------------- |
| `embedding`    | vector\[1536] | OpenAI/DeepSeek/Instructor LLM embeddings                                  |
| `content`      | str           | Full narrative: e.g., “In project X, removing `gender` improved R² by 10%” |
| `tags`         | list\[str]    | e.g., `["feature_selection", "credit_risk", "recall"]`                     |
| `source_agent` | str           | Which agent logged it                                                      |
| `timestamp`    | datetime      | Logged time                                                                |
| `project_id`   | str           | For filtering memory by session or context                                  |

This is used when agents perform semantic lookup, e.g.:
    - “Have we seen similar data before?”
    - "What cleaning actions worked for missing values in healthcare datasets?”

#### Symbolic Store (Structured Logs & Decisions)

| Table/File                      | Description                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------ |
| `past_projects.jsonl`           | Each row logs project metadata, dataset shape, user decisions, model success   |
| `user_preferences.json`         | Tracks feedback: approved/rejected models, column types, format annoyances     |
| `feature_history.parquet`       | Logs per-column behavior across past runs (e.g., always dropped, high leakage) |
| `model_performance_log.parquet` | Tracks models run, their params, and scores over time                          |
| `agent_interactions.log`        | Text logs of decisions, user overrides, and rationale                          |

#### What Each Agent Stores / Retrieves

| Agent              | Stores to Memory                                                         | Retrieves from Memory                                   |
| ------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------- |
| **Data Ingestion** | Dataset checksum, file metadata, target column, raw shape                 | Prior runs with same schema                             |
| **Data Profiling**  | Column summaries, data types, nulls, flagged columns                      | Patterns in similar datasets (e.g., outlier thresholds) |
| **Cleaning**       | What transformations were applied (e.g., imputation) + code snippets     | Common cleaning ops per column name/type                |
| **Feature Eng**    | Features generated and success indicators (e.g., correlation, SHAP gain) | Previously successful features from similar data        |
| **Modeling**       | Model configs, hyperparams, success scores                                | Best models for this target/data profile                 |
| **Evaluation**     | Metric interpretations, user feedback on reports                         | Thresholds used in past, favored metrics                |
| **Report Agent**   | Insight generation text chunks                                           | Style, format, successful past interpretations          |

### Retrieval Logic
Each agent uses a MemoryClient class with these methods:

```python
class MemoryClient:
    def store_vector(self, content: str, tags: List[str], source_agent: str, metadata: dict): ...
    def search_similar(self, query: str, top_k=5) -> List[Dict]: ...
    
    def store_symbolic(self, table_name: str, row_dict: dict): ...
    def query_symbolic(self, table_name: str, filters: dict) -> List[Dict]: ...
```

### Example: How Memory Affects Agent Behavior

**Feature Engineering Agent**
- Looks at current dataset profile
- Queries vector store:
    - "What features worked for binary classification tasks with low-cardinality categoricals?"
- Gets:
    - "income_per_household" improved AUC by 8% on similar census dataset. Feature: income / num_people"
- Suggests that feature to user

### Example: Learning from Feedback
User marks a report as “inaccurate interpretation.” The Evaluation Agent stores:
```json
{
  "feedback": "Insight missed the class imbalance issue.",
  "source_agent": "EvaluationAgent",
  "dataset_id": "xyz123",
  "tags": ["imbalance", "recall", "false_negative"]
}
```
Next time it sees a dataset with a similar issue, it retrieves that memory and adjusts its interpretation prompt accordingly.

## Human In The Loop Interactions
**System Principle**: Each phase ends with a checkpoint interface where the user can inspect the result, accept/reject, override, or rollback.

### Key User Capabilities
| Capability                    | Enabled? | Description                                                                                        |
| ----------------------------  | -------- | -------------------------------------------------------------------------------------------------- |
| ❌ Skip phase                 | ❌ No     | All phases are mandatory unless automatically skipped due to triviality (e.g., no missing values). |
| ✅ Review results             | ✅ Yes    | Users get a summary and visual output                                                              |
| ✅ Request rework             | ✅ Yes    | Agents rerun with user feedback                                                                    |
| ✅ Override decisions         | ✅ Yes    | User can choose model, features, remove rows, etc.                                                 |
| ✅ Upload custom code         | ✅ Yes    | User can paste or upload transformation snippets                                                   |
| ✅ Rollback to previous phase | ✅ Yes    | Reverts pipeline state and wipes affected downstream phases                                        |
| ✅ Save/Load sessions         | ✅ Yes    | User can save entire pipeline state and resume later                                               |

### Overridable Decisions per Phase
| Phase                   | Overridable Elements                                       |
| ----------------------- | ---------------------------------------------------------- |
| **Data Cleaning**       | Imputation method, column drop, type conversion            |
| **Feature Selection**   | Drop/include features manually                             |
| **Feature Engineering** | Approve/reject LLM suggestions, add new ones               |
| **Modeling**            | Algorithm selection (XGBoost vs. Logistic), scoring metric |
| **Evaluation**          | Thresholds, interpretation focus (recall/precision)        |
| **Report**              | Regenerate with different focus or wording tone            |

### Rollback Flow
- Each phase stores its pre- and post-execution state in `st.session_state["phase_snapshots"]`
- Rollback restores from the last approved checkpoint and clears downstream results.
```python
session_state["snapshots"]["feature_eng"] = {
    "input_data": df_before,
    "agent_output": df_after,
    "logs": logs
}
```

### User Interface Flow per Phase (Streamlit Logic)

**General Phase Control Panel**
──────────────────────────────────────
[Phase Title: Feature Engineering]
──────────────────────────────────────
Summary:
“Created 3 new features: log_income, age_bucket, income_to_expense_ratio”

Preview:
[DataFrame preview or chart]

Logs:
“Dropped ‘zipcode’ due to high cardinality and low variance”

-------------------------
User Actions:
-------------------------
Accept & Proceed

Rerun with:
   [ ] Remove feature: [input text]
   [ ] Add custom feature code:
       ```python
       df["new_feature"] = df["a"] / df["b"]
       ```

Suggestion: Use hashing encoder for 'city'

Rollback to Data Cleaning Phase
──────────────────────────────────────

### Rework Mechanism
When the user requests rework:
    - Feedback is passed to the current agent:

```python
feedback = {
    "drop_features": ["zipcode"],
    "add_code": "df['log_income'] = np.log(df['income'] + 1)"
}
```
Agent runs `.run(feedback=feedback)`

### LLM Prompt Refinement on Rework
If rework is requested:
    - The LLM prompt incorporates the feedback:
```python
# Prompt
"""
You are refining feature engineering on a dataset. The user has rejected the prior features because 'zipcode' was irrelevant. 

Instead, create meaningful features based on the columns 'income', 'expenses', and 'employment_status'.
"""
```

### Summary Flowchart: Human-in-the-Loop Interaction
[Phase Starts]
     │
     ▼
[Agent Runs → Produces Output]
     │
     ▼
[User Interface]
 ┌────────────┬────────────┬─────────────┐
 │ Accept     │ Request    │ Rollback    │
 │ Output     │ Rework     │ Phase       │
 └────────────┴────────────┴─────────────┘
     │              │             │
     ▼              ▼             ▼
Next Agent      Agent runs     Previous state
                with feedback   restored

## Modeling Policy

1. Supported Models (Shortlist Based on Task)
**For Classification**
| Model               | Use Case Guideline                              |
| ------------------- | ----------------------------------------------- |
| Logistic Regression | Baseline, interpretability                      |
| Random Forest       | Tabular data, non-linear, robust to noise       |
| XGBoost             | Benchmark model for structured classification   |
| K-Nearest Neighbors | Low dimensional, small datasets                 |
| Naive Bayes         | High-cardinality categorical, text-based inputs |
| SVM (Linear/RBF)    | Margin-focused, low-sample                      |

**For Regression**
| Model                   | Use Case Guideline                   |
| ----------------------- | ------------------------------------ |
| Linear Regression       | Baseline, explainability             |
| Ridge/Lasso             | Multicollinearity, feature shrinkage |
| Random Forest Regressor | Robust to outliers, nonlinear data   |
| XGBoost Regressor       | Best performance, can overfit        |
| SVR                     | Precise, slower on large datasets    |

**For Time Series Modeling
| Model           | Use Case Guideline                          |
| --------------- | ------------------------------------------- |
| ARIMA           | Univariate, linear trends                   |
| Prophet         | Seasonal + holiday patterns                 |
| XGBoost w/ Lags | Multivariate or engineered features         |
| LSTM (optional) | Only if data volume justifies it (deferred) |

2. Model Selection Framework (Best Practice)

Each model is evaluated across 3 dimensions:

**A. Dataset Characteristics**
| Factor                      | Influence                                 |
| --------------------------- | ----------------------------------------- |
| Number of rows              | Avoid deep models < 500 rows              |
| Feature count               | Use regularized models if > 100 features  |
| Cardinality of categoricals | Tree models > One-hot models              |
| Multicollinearity           | Use Lasso/Ridge                           |
| Missing values              | XGBoost tolerates it; linear models don’t |

**B. Task Objectives**
| Objective        | Preferred Models         |
| ---------------- | ------------------------ |
| Interpretability | Linear, Lasso, Logistic  |
| Accuracy         | XGBoost, RF              |
| Speed            | Logistic, Ridge          |
| Class imbalance  | Tree-based + AUC scoring |

**C. Metric Sensitivity**
| Metric Chosen | Bias Toward                        |
| ------------- | ---------------------------------- |
| F1, Recall    | Robust models with class weighting |
| ROC AUC       | Probabilistic classifiers          |
| R², MSE       | Balanced regression                |
| MAE           | Robust to outliers (RF, Lasso)     |

3. Hyperparameter Tuning Policy

**Strategy:**
    - Tier 1 (Light tuning):
        - Use `GridSearchCV` or `RandomizedSearchCV` on top 3 candidates
        - Search depth: up to 5 parameters × 5 values each
        - Time budget: 2–5 mins/model max (adjustable)
    - When to Tune:
        - Skip tuning for:
            - Small datasets (< 500 rows)
            - Early-stage prototyping
        - Tune when:
            - Model performance variance is high
            - Metric gaps are small across models

4. Ensemble Logic
| Ensemble Type          | Policy                             |
| ---------------------- | ---------------------------------- |
| **Bagging (RF)**       | Always considered                  |
| **Boosting (XGBoost)** | Strong default for structured data |
| **Stacking**           | Use for only the below reaons      |
| Voting                 | Used when individual model confidence is low |

- 3+ base models with complementary performance
- Ensemble yields ≥ 5% lift on validation metric

Evaluation:
- If no model outperforms baseline by >5%, an ensemble is tried
- Ensemble must show net improvement (validated) to be accepted

5. Resource & Execution Constraints
| Limit                  | Setting                       |
| ---------------------- | ----------------------------- |
| Max models per run     | 5 base models                 |
| Max time per model     | 5 minutes                     |
| Max hyperparam configs  | 25 per model                  |
| GPU usage              | ❌ Disabled (CPU only for now)|
| Memory limit           | 4GB per model job             |

6. Final Model Justification Report
The Modeling Agent must generate a rationale for the selected model:
```markdown
### Selected Model: XGBoostClassifier

**Why this model?**
- The dataset had 10k rows and 15 features with non-linear relationships.
- Class imbalance was present (70:30), and XGBoost handles this with built-in weighting.
- AUC was highest at 0.91 vs 0.83 (RandomForest) and 0.76 (LogisticRegression).

**Alternatives Considered:**
- Logistic Regression (low AUC, poor recall)
- Random Forest (better recall, lower precision)
- Stacking Ensemble (only 1% gain, not worth complexity)
```

## Time Series Modeling Policy
**1. Assumptions**
| Aspect                    | Assumption                                                        |
| ------------------------- | ----------------------------------------------------------------- |
| **Time Column**           | Required — must contain a column parsable as `datetime`           |
| **Forecast Target**       | One target variable only per run                                  |
| **Forecast Horizon**      | Default = **next 90 days** (user can override)                    |
| **Granularity**           | Auto-inferred (daily/weekly/monthly)                              |
| **Multivariate Support**  | ✅ Yes — multivariate forecasting supported                        |
| **Validation**            | ✅ Uses **walk-forward validation** (not just train/test split)    |
| **Missing Dates**         | Will be **auto-resampled** and filled based on inferred frequency |
| **Seasonality Detection** | Automatically inferred (with LLM insight support if needed)       |

**2. Preprocessing Policies (Best Practices)**
| Step                       | Logic                                                                              |
| -------------------------- | ---------------------------------------------------------------------------------- |
| **Datetime Parsing**       | Agent must auto-detect date column (or request user input)                         |
| **Resampling**             | Time index is resampled to uniform frequency (e.g., daily)                         |
| **Missing Timestamps**     | Filled using: forward-fill (`ffill`) → linear → interpolate                        |
| **Lag Feature Creation**   | By default, create lags: t-1 to t-7                                                |
| **Rolling Stats**          | Auto-generate: 7-day/30-day mean, std, min, max                                    |
| **Datetime Decomposition** | Extract day-of-week, month, year, is\_weekend                                      |
| **Train/Validation Split** | Walk-forward (e.g., sliding window) with final holdout set for evaluation          |
| **Stationarity Check**     | Perform ADF test; if non-stationary → log-diff or seasonal differencing (optional) |

All steps are logged and user-reviewable before modeling

**3. Supported Models & When to Use Them**
| Model                                | Use Case                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------- |
| **ARIMA**                            | Univariate, stationary data                                                           |
| **SARIMA**                           | Univariate + seasonal patterns                                                        |
| **Prophet**                          | Best for daily/monthly, interpretable, seasonal + holiday-aware                       |
| **XGBoost + Lags**                   | Best for multivariate with strong predictors                                          |
| **RandomForestRegressor + Features** | Non-linear multivariate                                                               |
| **LSTM (optional)**                  | Only enabled if user requests and dataset has 10k+ rows with long historical patterns |

**4. Forecast Horizon Control**
| Horizon Source       | Behavior                                                   |
| -------------------- | ---------------------------------------------------------- |
| None (default)       | Forecast **next 90 days**                                  |
| User-defined         | Accepts custom horizon in UI (e.g., next 30, 60, 180 days) |
| Inferred Seasonality | If weekly, 13-week horizon (quarter) is suggested          |

** 5. Validation Policy (Walk-forward)**
A. Default Configuration
    - Window Size: Last 12 months (or max 80% of history)
    - Rolling Horizon: 30 days
    - Step Size: 1 week

B. Metrics Tracked
| Metric | Used When                              |
| ------ | -------------------------------------- |
| MAE    | General                                |
| RMSE   | Outlier-sensitive                      |
| MAPE   | Percent-error focused                  |
| SMAPE  | Symmetric error – used in final report |

Each model’s prediction vs. actual is plotted for each validation split.

**6. Output Artifacts**
Each time-series run outputs:
    - preprocessed_data.csv
    - forecast_[model].csv
    - forecast_plot.png (with CI bands if available)
    - model.pkl
    - report.md(includes insights, model rationale, performance summary)

**7. LLM Usage in Time Series**
| Task                                     | LLM Role                            |
| ---------------------------------------- | ----------------------------------- |
| Explain seasonality or lag relationships | ✅ Yes                               |
| Justify model choice                     | ✅ Yes                               |
| Interpret anomalies or dips/spikes       | ✅ Yes                               |
| Build model?                             | ❌ No — model training is code-based |

**Example Flow**
1. User uploads CSV with `date`, `sales`, and other variables
2. TimeSeriesAgent:
    - Parses `date`
    - Resamples to daily frequency
    - Fills missing timestamps
    - Generates lag/rolling features
    - Detects seasonality
    - Runs Prophet and XGBoost
    - Evaluates via walk-forward
    - Chooses best model (e.g., Prophet)
3. Forecasts next 90 days
4. Generates evaluation plots, metrics
5. LLM writes explanation:
    “Sales show a 7-day seasonality, consistent with weekly retail cycles. Prophet captured this trend with an SMAPE of 8.3%, outperforming XGBoost at 12.1%...”











