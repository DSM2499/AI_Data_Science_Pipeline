# AI Data Science Pipeline

An intelligent, automated end-to-end data science pipeline with advanced AI assistance, natural language feedback processing, and optimized performance for production workloads.

## 🚀 Key Features

- **🤖 AI-Powered Agents**: 8 specialized agents handling each phase with LLM integration
- **💬 Natural Language Feedback**: Describe changes in plain English - AI understands and implements
- **🧠 Intelligent Memory**: Vector + symbolic storage learns from past projects and decisions  
- **⚡ Performance Optimized**: Parallel processing, caching, and optimized evaluations
- **🔄 Smart Categorical Handling**: Advanced encoding strategies for mixed data types
- **📊 Comprehensive Evaluation**: Enhanced metrics, visualizations, and cross-validation
- **📝 Auto Report Generation**: Professional markdown reports with insights and recommendations
- **🎯 Human-in-the-Loop**: Approval workflows with rollback and override capabilities
- **🔍 Real-time Monitoring**: Live pipeline status, execution times, and memory usage

## 🏗️ Architecture

### Pipeline Phases
1. **🔄 Data Ingestion** - Intelligent data validation and format detection
2. **📊 Data Profiling** - Advanced statistical analysis and quality assessment  
3. **🧹 Data Cleaning** - AI-guided cleaning with natural language feedback
4. **🎯 Feature Selection** - Smart feature elimination using statistical methods
5. **⚙️ Feature Engineering** - Automated feature creation with categorical encoding
6. **🤖 Modeling** - Multi-algorithm training with hyperparameter optimization
7. **📈 Evaluation** - Parallel evaluation with enhanced metrics and visualizations
8. **📄 Report Generation** - Professional reports with AI-generated insights

### Advanced Capabilities
- **🗣️ Natural Language Interface**: "Create a ratio of income to age" → Automatic implementation
- **🔧 Smart Categorical Processing**: Auto-detects and encodes categorical data (binary, label, frequency, hash)
- **⚡ Parallel Optimization**: Concurrent evaluation, plotting, and LLM processing  
- **💾 Persistent Memory**: Learns patterns across projects for better suggestions
- **🎮 Interactive Feedback**: Real-time approval with rollback and override options
- **📊 Performance Monitoring**: Detailed execution tracking and optimization metrics

## 📋 Requirements

- Python 3.9+
- 4GB+ RAM (8GB recommended)
- OpenAI API key (optional, for LLM features)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Data_Science_Agent
   ```

2. **Set up environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   make install
   # or
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your API keys
   nano .env
   ```

4. **Initialize the project**
   ```bash
   make setup
   ```

## 🚀 Quick Start

1. **Start the application**
   ```bash
   make run
   # or
   streamlit run main.py
   ```

2. **Open in browser**
   - Navigate to `http://localhost:8501`

3. **Upload your dataset**
   - Support formats: CSV, Excel, Parquet
   - Select target variable for supervised learning

4. **Follow the pipeline**
   - Review and approve each phase
   - Provide feedback and overrides as needed
   - Monitor progress in real-time

## 🎯 Usage Examples

### Basic Supervised Learning
1. Upload a CSV file with labeled data (supports categorical columns)
2. Select the target column  
3. Follow the automated pipeline with real-time feedback
4. Use natural language to modify: *"Skip polynomial features, they're too complex"*
5. Get optimized model with comprehensive evaluation and professional report

### Natural Language Feature Engineering
```
User: "I want to create a new feature that combines age and income as a ratio, and skip any polynomial features because they make the model too complex."

AI: ✅ Understanding: Create custom ratio feature, disable polynomial operations
→ Automatically generates: data['age_to_income_ratio'] = data['age'] / data['income']
→ Skips polynomial feature generation
→ Re-runs pipeline with changes
```

### Smart Categorical Data Handling
- **Auto-detection**: Identifies hidden categorical columns (mixed types, strings)
- **Intelligent encoding**: Binary (2 categories) → Binary encoding, Low cardinality (≤10) → Label encoding, High cardinality → Frequency encoding
- **Fallback strategies**: Hash encoding for problematic data, graceful column dropping when needed
- **Error prevention**: Validates mathematical operations before applying to encoded data

### Advanced Feedback & Control
- **Rollback**: *"Go back to the previous step"* → Automatic state restoration
- **Phase control**: *"Force restart from feature engineering"* → Pipeline jumps to specified phase  
- **Custom overrides**: Add your own Python code for specialized processing
- **Memory learning**: System remembers successful patterns for similar future datasets

## 🧠 Memory System

The system maintains two types of memory:

### Vector Memory (Semantic)
- Stores natural language descriptions of datasets, features, and results
- Enables semantic search for similar situations
- Powered by ChromaDB with embedding models

### Symbolic Memory (Structured)
- Stores structured data like model parameters, metrics, and decisions
- Enables precise queries and analytics
- Uses SQLite for fast access

## 🔧 Development

### Setup Development Environment
```bash
make install-dev
make setup
```

### Code Quality
```bash
make lint          # Run linting
make format        # Format code
make type-check    # Type checking
make test          # Run tests
make pre-commit    # Run all checks
```

### Adding New Agents
1. Inherit from `BaseAgent` or `LLMCapableAgent`
2. Implement required methods: `run()`, `validate_inputs()`
3. Add LLM prompt generation if using LLM features
4. Register with OrchestrationAgent
5. Add UI components if needed

### Project Structure
```
Data_Science_Agent/
├── agents/              # Agent implementations
├── memory/              # Memory system (vector + symbolic)
├── ui/                  # Streamlit UI components
├── utils/               # Shared utilities
├── tests/               # Test suite
├── project_output/      # Generated artifacts
├── config.py           # Configuration management
├── main.py             # Application entry point
└── requirements.txt    # Dependencies
```

## 📊 Monitoring & Performance

### Real-time Dashboard
- **Pipeline Progress**: Live phase tracking with percentage completion
- **Agent Performance**: Execution times, success rates, and optimization metrics
- **Memory Analytics**: Vector store entries, symbolic data, and system usage
- **LLM Statistics**: Token usage, response times, and cost tracking

### Performance Optimizations
- **Parallel Processing**: 3-4x faster evaluation and report generation
- **Smart Caching**: Pre-computed data summaries and efficient markdown generation
- **Optimized Evaluations**: Concurrent metric calculation, plotting, and analysis
- **Timeout Protection**: 30-second limits prevent hanging on slow LLM calls
- **Graceful Degradation**: Fallback strategies when services are unavailable

### Execution Tracking
- **Detailed Logs**: Comprehensive execution logs in `project_output/logs/`
- **Timing Metrics**: Per-agent execution times and bottleneck identification
- **Error Analytics**: Categorized error tracking with recovery suggestions
- **Audit Trails**: Complete history of user interactions and pipeline decisions

## 🔒 Security Considerations

- Environment variables for sensitive API keys
- Input validation and sanitization
- Sandboxed code execution (recommended for production)
- Audit trails for all user interactions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

### Development Guidelines
- Follow existing code style and patterns
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📝 License

MIT License - see LICENSE file for details.

## 🐛 Troubleshooting

### Common Issues

**Categorical Data Conversion Errors**
- ✅ **Fixed in v2.0**: Advanced categorical encoding prevents "Could not convert to numeric" errors
- System auto-detects and encodes categorical columns before mathematical operations
- Use natural language feedback: *"Skip polynomial features if you have categorical data"*

**Memory initialization errors**
- Check ChromaDB permissions and sufficient disk space
- Verify Python 3.9+ compatibility
- Run `make setup` to reinitialize memory stores

**LLM integration issues**
- Verify API keys in `.env` file and check internet connectivity
- System includes timeout protection and graceful fallback
- LLM features are optional - pipeline works without them

**Performance Issues**
- **v2.0 Optimization**: 3-4x faster execution with parallel processing
- Large datasets automatically use optimized processing paths
- Monitor real-time performance metrics in the UI dashboard

### Error Recovery & Feedback

**Natural Language Error Resolution**
```
Error: "Feature engineering failed with categorical data"
Solution: "Skip operations that don't work with categorical columns and focus on numeric features only"
→ AI automatically adjusts pipeline and retries
```

**Pipeline State Management**
- Use rollback functionality for problematic phases
- Override specific operations through the UI
- Reset entire pipeline if needed: `🔄 Reset Pipeline` button

### Getting Help

1. **Check Real-time Logs**: View detailed execution logs in the UI sidebar
2. **Memory System Status**: Review vector and symbolic store health in dashboard  
3. **Performance Metrics**: Monitor agent execution times and success rates
4. **Natural Language Debug**: Describe issues in plain English for AI assistance
5. **GitHub Issues**: Report bugs with execution logs and performance metrics

## 🔮 Roadmap

### v2.1 (Next Release)
- [ ] **Time Series Agent**: Specialized forecasting with seasonal decomposition
- [ ] **Advanced AutoML**: Automated feature engineering with genetic algorithms  
- [ ] **Model Ensembling**: Intelligent stacking and blending strategies
- [ ] **Enhanced Visualizations**: Interactive plots and model explanations

### v3.0 (Future)
- [ ] **Real-time Inference**: Production-ready ML serving pipeline
- [ ] **Cloud Integration**: AWS, GCP, Azure deployment support
- [ ] **Multi-user Collaboration**: Shared projects and team workflows
- [ ] **Advanced NLP**: Document processing and text analytics agents

### Completed in v2.0 ✅
- [x] **Natural Language Feedback Processing**: Plain English pipeline control
- [x] **Advanced Categorical Encoding**: Smart detection and encoding strategies
- [x] **Performance Optimization**: 3-4x faster evaluation and report generation
- [x] **Enhanced Error Handling**: Robust categorical data processing
- [x] **Parallel Processing**: Concurrent agent execution and LLM calls
- [x] **Professional Reporting**: AI-generated insights and recommendations

## 📚 Documentation

- [Agent Specifications](Agent_spec.md)
- [Project Specifications](Project_spec.md)
- [CLAUDE.md](CLAUDE.md) - Development guide for Claude Code

---

## 🎉 What's New in v2.0

### 🗣️ Natural Language AI Control
Transform how you interact with the pipeline:
```
"Create a ratio of income to age and skip polynomial features"
→ AI understands and implements automatically
```

### 🔧 Smart Categorical Data Processing  
No more conversion errors:
- Auto-detects mixed data types and categorical columns
- Intelligent encoding: binary → label → frequency → hash fallback
- Mathematical operation validation before processing

### ⚡ 3-4x Performance Boost
- Parallel evaluation, plotting, and LLM processing
- Optimized report generation with concurrent sections
- Smart caching and pre-computed summaries

### 📊 Enhanced Analytics
- Comprehensive evaluation metrics (AUC-ROC, residual analysis)
- Professional report generation with AI insights
- Real-time performance monitoring and optimization

---
