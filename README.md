# AI Data Science Pipeline

An intelligent, automated data science pipeline with human-in-the-loop control, long-term memory, and learning capabilities.

## 🚀 Features

- **Multi-Agent Architecture**: Specialized agents for each phase of the data science pipeline
- **Human-in-the-Loop**: User approval and feedback at critical decision points
- **Memory System**: Learns from past decisions and datasets using vector + symbolic storage
- **LLM Integration**: Intelligent insights and explanations throughout the process
- **Interactive UI**: Streamlit-based interface for easy interaction and monitoring
- **Comprehensive Logging**: Detailed execution logs and audit trails
- **Rollback Support**: Ability to revert to previous pipeline states

## 🏗️ Architecture

### Pipeline Phases
1. **Data Ingestion** - Validate and load structured data files
2. **Data Profiling** - Generate statistical summaries and quality insights
3. **Data Cleaning** - Automated cleaning with user oversight
4. **Feature Selection** - Remove redundant and low-signal features
5. **Feature Engineering** - Create new features with AI assistance
6. **Modeling** - Train and evaluate multiple ML models
7. **Evaluation** - Comprehensive model assessment and interpretation
8. **Report Generation** - Automated insights and recommendations

### Core Components
- **OrchestrationAgent**: Central controller managing pipeline execution
- **Memory System**: ChromaDB for semantic search + SQLite for structured data
- **Agent Framework**: Extensible base classes for consistent agent behavior
- **UI Components**: Reusable Streamlit components for rich interactions

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
1. Upload a CSV file with labeled data
2. Select the target column
3. Follow the automated pipeline
4. Review and approve each phase
5. Get trained model and insights

### Custom Data Cleaning
1. Upload dataset and proceed to data cleaning
2. Review suggested cleaning operations
3. Add custom cleaning code if needed
4. Approve and continue to modeling

### Learning from History
- The system learns from your decisions
- Similar datasets get better default suggestions
- Past successful features are recommended
- Model performance patterns inform choices

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

## 📊 Monitoring & Logging

- **Execution Logs**: Detailed logs in `project_output/logs/`
- **Memory Stats**: Track memory usage and entries
- **Pipeline Status**: Real-time progress monitoring
- **Agent Performance**: Execution times and success rates

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

**Memory initialization errors**
- Check ChromaDB permissions
- Ensure sufficient disk space
- Verify Python version compatibility

**LLM integration issues**
- Verify API keys in .env file
- Check internet connectivity
- Review API rate limits

**Large dataset performance**
- Consider data sampling for exploration
- Monitor memory usage
- Use appropriate chunk sizes

### Getting Help

1. Check the logs in `project_output/logs/`
2. Review memory system status in the UI
3. Try resetting the pipeline
4. Check GitHub issues for known problems

## 🔮 Roadmap

- [ ] Time series forecasting agent
- [ ] Advanced feature engineering with AutoML
- [ ] Model ensemble and stacking
- [ ] Real-time inference pipeline
- [ ] Integration with cloud platforms
- [ ] Advanced visualization components
- [ ] Multi-user support and collaboration

## 📚 Documentation

- [Agent Specifications](Agent_spec.md)
- [Project Specifications](Project_spec.md)
- [CLAUDE.md](CLAUDE.md) - Development guide for Claude Code

---

**Built with ❤️ for the data science community**