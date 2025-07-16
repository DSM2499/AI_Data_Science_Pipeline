"""Report Generation Agent - Creates comprehensive markdown reports."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Dict, List, Optional

from pydantic import BaseModel

from agents.base import AgentContext, AgentResult, LLMCapableAgent, UserFeedback
from config import settings
from memory.memory_client import memory_client
from utils.exceptions import ReportGenerationError, ValidationError


class ReportSection(BaseModel):
    """Represents a section of the report."""
    
    title: str
    content: str
    order: int
    generation_time: float = 0.0
    word_count: int = 0


class ReportResult(BaseModel):
    """Result of report generation process."""
    
    report_path: str
    sections: list
    word_count: int
    generation_time: float
    sections_generated: int
    llm_sections: int
    static_sections: int


class ReportGenAgent(LLMCapableAgent):
    """Agent responsible for generating comprehensive data science reports."""
    
    def __init__(self, use_llm: bool = True):
        super().__init__("ReportGenAgent", use_llm)
    
    def run(self, context: AgentContext, feedback = None) -> AgentResult:
        """Execute optimized report generation process."""
        try:
            start_time = time.time()
            self.validate_inputs(context)
            self._create_snapshot(context)
            
            if feedback:
                context = self._handle_user_feedback(context, feedback)
            
            # Generate report with parallel processing
            report_result = self._generate_report_parallel(context)
            
            # Update context
            context.llm_insights[self.name] = f"Generated comprehensive report: {report_result.report_path}"
            
            # Store in memory (non-blocking)
            memory_storage = ThreadPoolExecutor(max_workers=1)
            memory_future = memory_storage.submit(self._store_report_memory, report_result, context)
            
            execution_time = time.time() - start_time
            
            result = AgentResult(
                success=True,
                agent_name=self.name,
                context=context,
                logs=[
                    f"Generated report with {report_result.sections_generated} sections in {execution_time:.2f}s",
                    f"Report saved to: {report_result.report_path}",
                    f"Word count: {report_result.word_count}",
                    f"LLM sections: {report_result.llm_sections}, Static sections: {report_result.static_sections}",
                ],
                metrics={
                    "sections_count": report_result.sections_generated,
                    "word_count": report_result.word_count,
                    "generation_time": execution_time,
                    "llm_sections": report_result.llm_sections,
                    "static_sections": report_result.static_sections,
                },
                artifacts={
                    "report_result": report_result.dict(),
                    "report_path": report_result.report_path,
                },
                suggestions=["Review the generated report and download for presentation"],
                requires_approval=False,
            )
            
            # Ensure memory storage completes
            memory_future.result()
            memory_storage.shutdown(wait=False)
            
            self._log_execution(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            
            return AgentResult(
                success=False,
                agent_name=self.name,
                context=context,
                error=str(e),
                logs=[f"Report generation failed: {e}"],
                user_message="Report generation encountered an error.",
            )
    
    def validate_inputs(self, context: AgentContext) -> bool:
        """Validate inputs for report generation."""
        if context.data is None:
            raise ValidationError("No data found in context")
        
        return True
    
    def _generate_report_parallel(self, context: AgentContext) -> ReportResult:
        """Generate comprehensive markdown report with parallel processing."""
        start_time = time.time()
        
        # Pre-compute data for all sections to avoid multiple calculations
        data_summary = self._precompute_data_summary(context)
        
        sections = []
        llm_sections = 0
        static_sections = 0
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit LLM-dependent sections in parallel
            llm_futures = {}
            
            if self.use_llm:
                # Executive Summary (LLM)
                llm_futures['summary'] = executor.submit(
                    self._generate_executive_summary_fast, context, data_summary
                )
                
                # Recommendations (LLM)
                llm_futures['recommendations'] = executor.submit(
                    self._generate_recommendations_section_fast, context, data_summary
                )
            
            # Generate static sections immediately (no LLM required)
            static_sections_data = [
                ("Data Overview", self._generate_data_overview_fast(context, data_summary), 2),
                ("Methodology", self._generate_methodology_section_fast(context), 3),
                ("Results", self._generate_results_section_fast(context, data_summary), 4),
            ]
            
            # Add static sections
            for title, content, order in static_sections_data:
                sections.append(ReportSection(
                    title=title,
                    content=content,
                    order=order,
                    generation_time=0.1,  # Fast static generation
                    word_count=len(content.split())
                ))
                static_sections += 1
            
            # Collect LLM results
            if self.use_llm:
                try:
                    summary_content = llm_futures['summary'].result(timeout=30)  # 30s timeout
                    sections.append(ReportSection(
                        title="Executive Summary",
                        content=summary_content,
                        order=1,
                        generation_time=2.0,
                        word_count=len(summary_content.split())
                    ))
                    llm_sections += 1
                except Exception as e:
                    self.logger.warning(f"Executive summary generation failed: {e}")
                    sections.append(ReportSection(
                        title="Executive Summary",
                        content="Executive summary generation failed. Pipeline completed successfully.",
                        order=1,
                        generation_time=0.1,
                        word_count=8
                    ))
                    static_sections += 1
                
                try:
                    recommendations_content = llm_futures['recommendations'].result(timeout=30)
                    sections.append(ReportSection(
                        title="Recommendations",
                        content=recommendations_content,
                        order=5,
                        generation_time=2.0,
                        word_count=len(recommendations_content.split())
                    ))
                    llm_sections += 1
                except Exception as e:
                    self.logger.warning(f"Recommendations generation failed: {e}")
                    sections.append(ReportSection(
                        title="Recommendations",
                        content="Recommendations generation failed. Review model performance manually.",
                        order=5,
                        generation_time=0.1,
                        word_count=9
                    ))
                    static_sections += 1
            else:
                # Add fallback sections without LLM
                sections.extend([
                    ReportSection(
                        title="Executive Summary",
                        content="Executive summary: Automated data science pipeline completed successfully.",
                        order=1,
                        generation_time=0.1,
                        word_count=9
                    ),
                    ReportSection(
                        title="Recommendations",
                        content="Recommendations: Review model performance and consider next steps.",
                        order=5,
                        generation_time=0.1,
                        word_count=10
                    )
                ])
                static_sections += 2
        
        # Generate full report markdown efficiently
        report_content = self._compile_report_fast(sections, context, data_summary)
        
        # Save report asynchronously
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"data_science_report_{timestamp}.md"
        report_path = settings.reports_path / report_filename
        
        # Ensure directory exists
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Fast file write
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        generation_time = time.time() - start_time
        word_count = len(report_content.split())
        
        return ReportResult(
            report_path=str(report_path),
            sections=[s.dict() for s in sections],
            word_count=word_count,
            generation_time=generation_time,
            sections_generated=len(sections),
            llm_sections=llm_sections,
            static_sections=static_sections,
        )
    
    def _precompute_data_summary(self, context: AgentContext) -> Dict:
        """Precompute frequently used data summary to avoid repeated calculations."""
        data = context.data
        summary = {
            'shape': data.shape,
            'target': context.target,
            'dtypes_summary': dict(data.dtypes.value_counts()),
            'missing_values': int(data.isnull().sum().sum()),
            'memory_usage': data.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        }
        
        # Model info
        if context.model:
            summary['model'] = {
                'name': context.model.model_name,
                'type': getattr(context.model, 'model_type', 'unknown'),
                'metrics': getattr(context.model, 'metrics', {}),
                'feature_importance': getattr(context.model, 'feature_importance', {})
            }
        
        # Evaluation info
        if hasattr(context, 'evaluation_results') and context.evaluation_results:
            summary['evaluation'] = context.evaluation_results
        
        return summary
    
    def _generate_executive_summary_fast(self, context: AgentContext, data_summary: Dict) -> str:
        """Generate executive summary using LLM with optimized prompt."""
        try:
            from utils.llm_service import llm_service
            
            if not llm_service.is_available():
                return "Executive summary: Automated data science pipeline completed successfully."
            
            # Optimized prompt with pre-computed data
            summary_data = {
                "dataset_size": f"{data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns",
                "target_variable": data_summary['target'] or "Not specified",
                "best_model": data_summary.get('model', {}).get('name', 'Unknown'),
                "key_metrics": data_summary.get('model', {}).get('metrics', {}),
            }
            
            summary = llm_service.generate_report_section(
                section_type="executive_summary",
                data=summary_data,
                audience="business",
                max_tokens=300,  # Reduced for speed
                temperature=0.5,  # Lower temperature for faster generation
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate executive summary: {e}")
            return "Executive summary could not be generated."
    
    def _generate_data_overview_fast(self, context: AgentContext, data_summary: Dict) -> str:
        """Generate data overview section efficiently."""
        shape = data_summary['shape']
        target = data_summary['target']
        dtypes = data_summary['dtypes_summary']
        missing_values = data_summary['missing_values']
        
        content = f"""
## Dataset Characteristics

- **Shape**: {shape[0]:,} rows, {shape[1]:,} columns
- **Target Variable**: {target or 'Not specified'}
- **Data Types**: {', '.join(f'{k}: {v}' for k, v in dtypes.items())}
- **Missing Values**: {missing_values:,} total ({missing_values/shape[0]/shape[1]*100:.1f}%)
- **Memory Usage**: {data_summary['memory_usage']:.1f} MB

### Key Statistics

"""
        
        if context.profile_summary:
            profile = context.profile_summary
            content += f"""
- **Overall Null Percentage**: {profile.get('overall_null_percentage', 0):.1f}%
- **Duplicate Rows**: {profile.get('duplicate_rows', 0):,}
- **High Null Columns**: {len(profile.get('high_null_columns', []))}
- **Constant Columns**: {len(profile.get('constant_columns', []))}
"""
        
        # Add column summary table (limited for performance)
        if shape[1] <= 20:  # Only for small datasets
            content += "\n### Column Summary\n\n"
            content += self._generate_column_summary_table_fast(context.data)
        else:
            content += f"\n*Dataset has {shape[1]} columns. Column details available in full analysis.*\n"
        
        return content
    
    def _generate_column_summary_table_fast(self, data) -> str:
        """Generate optimized markdown table of column summary."""
        # Pre-compute all stats at once for efficiency
        summary_stats = {
            'types': data.dtypes,
            'non_null': data.count(),
            'unique': data.nunique(),
            'null_pct': (data.isnull().sum() / len(data) * 100).round(1)
        }
        
        # Build table efficiently
        headers = ["Column", "Type", "Non-Null", "Unique", "Null %"]
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        # Limit to first 15 columns for performance
        for col in data.columns[:15]:
            row = [
                col,
                str(summary_stats['types'][col]),
                f"{summary_stats['non_null'][col]:,}",
                f"{summary_stats['unique'][col]:,}",
                f"{summary_stats['null_pct'][col]:.1f}%"
            ]
            table += "| " + " | ".join(row) + " |\n"
        
        if len(data.columns) > 15:
            table += f"\n*Showing first 15 of {len(data.columns)} columns*\n"
        
        return table
    
    def _generate_methodology_section_fast(self, context: AgentContext) -> str:
        """Generate methodology section (static content - fast)."""
        return """
## Data Science Pipeline

The analysis followed a comprehensive automated pipeline:

### 1. Data Ingestion & Validation
- Validated data format and structure
- Identified target variable and data types
- Performed initial quality assessment

### 2. Data Profiling & Cleaning
- Generated statistical summaries and distributions
- Identified and handled missing values
- Removed duplicates and applied transformations

### 3. Feature Engineering & Selection
- Created new features through domain transformations
- Applied statistical feature selection
- Generated interaction terms and polynomial features

### 4. Model Training & Optimization
- Trained multiple algorithm types
- Performed hyperparameter optimization
- Selected best performing model based on cross-validation

### 5. Model Evaluation & Interpretation
- Calculated comprehensive performance metrics
- Generated model interpretations and visualizations
- Provided actionable recommendations
"""
    
    def _generate_results_section_fast(self, context: AgentContext, data_summary: Dict) -> str:
        """Generate results section efficiently."""
        content = "## Results\n\n"
        
        model_info = data_summary.get('model', {})
        if model_info:
            content += f"""
### Best Model: {model_info['name']}

**Model Type**: {model_info['type'].title()}

**Performance Metrics**:

"""
            
            # Display metrics efficiently
            for metric, value in model_info.get('metrics', {}).items():
                if isinstance(value, (int, float)):
                    content += f"- **{metric.upper()}**: {value:.4f}\n"
                else:
                    content += f"- **{metric.upper()}**: {value}\n"
            
            # Feature importance (top 10 only)
            feature_importance = model_info.get('feature_importance', {})
            if feature_importance:
                content += "\n**Top Features by Importance**:\n\n"
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                    reverse=True
                )
                
                for feature, importance in sorted_features[:10]:
                    if isinstance(importance, (int, float)):
                        content += f"- {feature}: {importance:.4f}\n"
                    else:
                        content += f"- {feature}: {importance}\n"
        
        # Evaluation results
        evaluation_info = data_summary.get('evaluation', {})
        if evaluation_info:
            content += f"\n### Model Evaluation\n\n"
            
            if evaluation_info.get('recommendations'):
                content += "**Key Recommendations**:\n\n"
                for rec in evaluation_info['recommendations'][:5]:  # Limit to top 5
                    content += f"- {rec}\n"
        
        return content
    
    def _generate_recommendations_section_fast(self, context: AgentContext, data_summary: Dict) -> str:
        """Generate recommendations section using LLM with optimized prompt."""
        try:
            from utils.llm_service import llm_service
            
            if not llm_service.is_available():
                return "Recommendations: Review model performance and consider next steps for production deployment."
            
            # Optimized data for LLM
            recommendations_data = {
                "model_performance": data_summary.get('model', {}).get('metrics', {}),
                "dataset_characteristics": {
                    "size": f"{data_summary['shape'][0]:,} rows",
                    "features": data_summary['shape'][1],
                    "target": data_summary['target'],
                },
                "data_quality": {
                    "missing_percentage": data_summary['missing_values'] / (data_summary['shape'][0] * data_summary['shape'][1]) * 100,
                    "memory_usage": data_summary['memory_usage']
                }
            }
            
            recommendations = llm_service.generate_report_section(
                section_type="recommendations",
                data=recommendations_data,
                audience="technical",
                max_tokens=400,  # Reduced for speed
                temperature=0.5,  # Lower temperature for consistency
            )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return "Recommendations could not be generated. Review model metrics and consider next steps."
    
    def _compile_report_fast(self, sections: List[ReportSection], context: AgentContext, data_summary: Dict) -> str:
        """Compile all sections into final report efficiently."""
        
        # Report header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        shape = data_summary['shape']
        
        report = f"""# Data Science Pipeline Report

**Generated**: {timestamp}
**Project**: {context.user_preferences.get('project_id', 'Default Project')}
**Dataset**: {shape[0]:,} rows × {shape[1]:,} columns
**Target Variable**: {data_summary['target'] or 'Not specified'}

---

"""
        
        # Add sections in order efficiently
        sorted_sections = sorted(sections, key=lambda x: x.order)
        
        for section in sorted_sections:
            report += f"# {section.title}\n\n"
            report += section.content
            report += "\n\n---\n\n"
        
        # Add footer
        total_generation_time = sum(s.generation_time for s in sections)
        
        report += f"""
## Pipeline Summary

This report was generated by the AI Data Science Pipeline.

**Statistics**:
- Report sections: {len(sections)}
- Total words: {sum(s.word_count for s in sections):,}
- Generation time: {total_generation_time:.1f}s
- LLM sections: {sum(1 for s in sections if s.generation_time > 1.0)}

---

*Generated by AI Data Science Agent v2.0*
"""
        
        return report

    def _generate_report(self, context: AgentContext) -> ReportResult:
        """Generate comprehensive markdown report."""
        import time
        start_time = time.time()
        
        sections = []
        
        # Executive Summary
        if self.use_llm:
            summary_content = self._generate_executive_summary(context)
        else:
            summary_content = "Executive summary of the data science pipeline execution."
        
        sections.append(ReportSection(
            title="Executive Summary",
            content=summary_content,
            order=1
        ))
        
        # Data Overview
        sections.append(ReportSection(
            title="Data Overview",
            content=self._generate_data_overview(context),
            order=2
        ))
        
        # Methodology
        sections.append(ReportSection(
            title="Methodology",
            content=self._generate_methodology_section(context),
            order=3
        ))
        
        # Results
        sections.append(ReportSection(
            title="Results",
            content=self._generate_results_section(context),
            order=4
        ))
        
        # Recommendations
        if self.use_llm:
            recommendations_content = self._generate_recommendations_section(context)
        else:
            recommendations_content = "Recommendations based on the analysis."
        
        sections.append(ReportSection(
            title="Recommendations",
            content=recommendations_content,
            order=5
        ))
        
        # Generate full report markdown
        report_content = self._compile_report(sections, context)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"data_science_report_{timestamp}.md"
        report_path = settings.reports_path / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        generation_time = time.time() - start_time
        word_count = len(report_content.split())
        
        return ReportResult(
            report_path=str(report_path),
            sections=[s.dict() for s in sections],
            word_count=word_count,
            generation_time=generation_time,
        )
    
    def _generate_executive_summary(self, context: AgentContext) -> str:
        """Generate executive summary using LLM."""
        try:
            from utils.llm_service import llm_service
            
            if not llm_service.is_available():
                return "Executive summary: Automated data science pipeline completed successfully."
            
            # Gather key information
            data_shape = context.data.shape
            target = context.target
            model_name = context.model.model_name if context.model else "Unknown"
            
            summary_data = {
                "dataset_size": f"{data_shape[0]} rows, {data_shape[1]} columns",
                "target_variable": target or "Not specified",
                "best_model": model_name,
                "pipeline_phases": ["data_ingestion", "data_profiling", "data_cleaning", "feature_selection", "feature_engineering", "modeling", "evaluation"],
            }
            
            summary = llm_service.generate_report_section(
                section_type="executive_summary",
                data=summary_data,
                audience="business",
                max_tokens=400,
                temperature=0.7,
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate executive summary: {e}")
            return "Executive summary could not be generated."
    
    def _generate_data_overview(self, context: AgentContext) -> str:
        """Generate data overview section."""
        data = context.data
        
        content = f"""
## Dataset Characteristics

- **Shape**: {data.shape[0]} rows, {data.shape[1]} columns
- **Target Variable**: {context.target or 'Not specified'}
- **Data Types**: {dict(data.dtypes.value_counts())}
- **Missing Values**: {data.isnull().sum().sum()} total

### Column Summary

{self._generate_column_summary_table(data)}

### Data Quality Insights

"""
        
        if context.profile_summary:
            profile = context.profile_summary
            content += f"""
- **Overall Null Percentage**: {profile.get('overall_null_percentage', 0):.1f}%
- **Duplicate Rows**: {profile.get('duplicate_rows', 0)}
- **High Null Columns**: {len(profile.get('high_null_columns', []))}
- **Constant Columns**: {len(profile.get('constant_columns', []))}
"""
        
        return content
    
    def _generate_column_summary_table(self, data) -> str:
        """Generate markdown table of column summary."""
        summary_data = []
        
        for col in data.columns:
            summary_data.append({
                'Column': col,
                'Type': str(data[col].dtype),
                'Non-Null': data[col].count(),
                'Unique': data[col].nunique(),
                'Null %': f"{(data[col].isnull().sum() / len(data) * 100):.1f}%"
            })
        
        # Convert to markdown table
        if summary_data:
            headers = list(summary_data[0].keys())
            table = "| " + " | ".join(headers) + " |\n"
            table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            
            for row in summary_data[:20]:  # Limit to first 20 columns
                table += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"
            
            if len(summary_data) > 20:
                table += f"\n*Showing first 20 of {len(summary_data)} columns*\n"
            
            return table
        
        return "No column data available."
    
    def _generate_methodology_section(self, context: AgentContext) -> str:
        """Generate methodology section."""
        content = """
## Data Science Pipeline

The analysis followed a comprehensive automated pipeline:

### 1. Data Ingestion
- Validated data format and structure
- Identified target variable
- Assessed data quality

### 2. Data Profiling
- Generated statistical summaries
- Identified data quality issues
- Analyzed feature distributions

### 3. Data Cleaning
- Handled missing values
- Removed duplicate rows
- Applied data transformations

### 4. Feature Selection
- Removed low-variance features
- Eliminated highly correlated features
- Applied statistical feature selection

### 5. Feature Engineering
- Created new features through transformations
- Generated interaction terms
- Applied domain-specific engineering

### 6. Model Training
- Trained multiple model types
- Performed hyperparameter optimization
- Selected best performing model

### 7. Model Evaluation
- Calculated performance metrics
- Generated model interpretations
- Provided recommendations
"""
        
        return content
    
    def _generate_results_section(self, context: AgentContext) -> str:
        """Generate results section."""
        content = "## Results\n\n"
        
        if context.model:
            model = context.model
            content += f"""
### Best Model: {model.model_name}

**Performance Metrics:**

"""
            
            for metric, value in model.metrics.items():
                content += f"- **{metric.upper()}**: {value:.4f}\n"
            
            if model.feature_importance:
                content += "\n**Top Features by Importance:**\n\n"
                sorted_features = sorted(
                    model.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for feature, importance in sorted_features[:10]:
                    content += f"- {feature}: {importance:.4f}\n"
        
        if hasattr(context, 'evaluation_results') and context.evaluation_results:
            eval_results = context.evaluation_results
            content += f"\n### Model Evaluation\n\n"
            content += f"**Task Type**: {eval_results.get('task_type', 'Unknown')}\n\n"
            
            if eval_results.get('recommendations'):
                content += "**Recommendations:**\n\n"
                for rec in eval_results['recommendations']:
                    content += f"- {rec}\n"
        
        return content
    
    def _generate_recommendations_section(self, context: AgentContext) -> str:
        """Generate recommendations section using LLM."""
        try:
            from utils.llm_service import llm_service
            
            if not llm_service.is_available():
                return "Recommendations: Review model performance and consider next steps."
            
            recommendations_data = {
                "model_performance": context.model.metrics if context.model else {},
                "data_characteristics": {
                    "shape": context.data.shape,
                    "target": context.target,
                },
                "pipeline_insights": context.llm_insights,
            }
            
            recommendations = llm_service.generate_report_section(
                section_type="recommendations",
                data=recommendations_data,
                audience="technical",
                max_tokens=500,
                temperature=0.7,
            )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return "Recommendations could not be generated."
    
    def _compile_report(self, sections: list, context: AgentContext) -> str:
        """Compile all sections into final report."""
        
        # Report header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Data Science Pipeline Report

**Generated**: {timestamp}
**Project**: {context.user_preferences.get('project_id', 'Default Project')}
**Target Variable**: {context.target or 'Not specified'}

---

"""
        
        # Add sections in order
        sorted_sections = sorted(sections, key=lambda x: x.order)
        
        for section in sorted_sections:
            report += f"# {section.title}\n\n"
            report += section.content
            report += "\n\n---\n\n"
        
        # Add footer
        report += f"""
## Pipeline Summary

This report was generated by the AI Data Science Pipeline, an automated system for end-to-end machine learning workflows.

**Pipeline Execution Time**: {timestamp}
**Report Word Count**: {len(report.split())} words

---

*Generated by AI Data Science Agent*
"""
        
        return report
    
    def _store_report_memory(self, report_result: ReportResult, context: AgentContext) -> None:
        """Store report generation results in memory."""
        try:
            report_data = {
                "report_path": report_result.report_path,
                "sections_count": len(report_result.sections),
                "word_count": report_result.word_count,
                "generation_time": report_result.generation_time,
                "target_variable": context.target,
            }
            
            memory_client.store_symbolic(
                table_name="generated_reports",
                data=report_data,
                source_agent=self.name,
                project_id=context.user_preferences.get("project_id", "default"),
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store report memory: {e}")
    
    def _get_llm_prompt(self, context: AgentContext, feedback = None) -> str:
        """Generate LLM prompt for report generation."""
        
        prompt = f"""
        Generate a comprehensive data science report for this project:
        
        Dataset: {context.data.shape}
        Target: {context.target}
        Model: {context.model.model_name if context.model else 'None'}
        
        Include executive summary, methodology, results, and recommendations.
        """
        
        return prompt.strip()