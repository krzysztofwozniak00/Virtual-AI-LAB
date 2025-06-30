# Enhanced R&D Workflow Evaluation System

## Overview

This enhanced evaluation system provides comprehensive quality metrics for your R&D multi-agent workflow, including:

- **Per-Agent Quality Scores**: Individual performance metrics for each agent
- **Workflow Success Metrics**: 0-7 score based on output generation
- **End-to-End Quality**: Overall workflow effectiveness
- **Model Comparison Tools**: Compare GPT-4o, GPT-4-turbo, GPT-4o-mini, O1-preview, O1-mini
- **Parameter Studies**: Temperature and token limit impact analysis

## New Evaluation Metrics

### Per-Agent Quality Evaluators

1. **Research Agent Quality** (0-100%)
   - Market analysis depth
   - Technical research quality
   - Source diversity
   - Strategic insights
   - Structure & clarity

2. **Ideation Agent Quality** (0-100%)
   - Innovation level
   - Concept diversity
   - Feasibility assessment
   - Strategic alignment
   - Concept completeness

3. **Evaluation Agent Quality** (0-100%)
   - Selection rationale
   - Strategic analysis depth
   - Implementation planning
   - Financial considerations
   - Risk assessment quality

4. **Specification Agent Quality** (0-100%)
   - Technical completeness
   - Implementation detail
   - Architecture quality
   - Standards compliance
   - Documentation clarity

5. **Testing Agent Quality** (0-100%)
   - Test coverage
   - Validation methodology
   - Quality metrics
   - Risk-based testing
   - Compliance & standards

### Overall Workflow Metrics

- **Workflow Success v2**: Counts outputs (0-7), each output = 1/7 of score
- **End-to-End Quality**: Overall request fulfillment and coherence

## Quick Start

### 1. Setup Enhanced Evaluation

```bash
# Replace old evaluation file
mv agent_lab_evals.py agent_lab_evals_old.py
mv agent_lab_evals_enhanced.py agent_lab_evals.py

# Or use the enhanced version directly by updating imports in main.py
```

### 2. Create Persistent Dataset (Once)

```bash
python create_dataset.py
```

### 3. Run Quick Evaluations

```bash
# Interactive quick evaluation menu
python run_quick_eval.py

# Options:
# 1. Baseline (GPT-4o standard)
# 2. Quick Model Comparison
# 3. Temperature Impact Test
# 4. All Tests
```

### 4. Run Specific Experiments

```bash
# GPT-4o Baseline
python run_experiment_cli.py --model gpt-4o --temperature 0.7 --max-tokens 8000 --name "baseline"

# Compare Models
python run_experiment_cli.py --compare gpt-4o gpt-4-turbo gpt-4o-mini o1-mini

# Temperature Study
python run_experiment_cli.py --model gpt-4o --temperature 0.1 --name "low_temp"
python run_experiment_cli.py --model gpt-4o --temperature 0.9 --name "high_temp"

# Token Study
python run_experiment_cli.py --model gpt-4o --max-tokens 2000 --name "small_context"
python run_experiment_cli.py --model gpt-4o --max-tokens 16000 --name "large_context"
```

### 5. Comprehensive Comparison Suite

```bash
python run_model_comparison.py

# Select:
# 2 - Model comparison (all models at fixed settings)
# 3 - Temperature study (0.1 to 0.9)
# 4 - Token study (2k to 16k)
# 5 - Comprehensive matrix (all combinations)
```

## Model Configurations

### Standard Models
- **GPT-4o**: Best overall, supports temp 0-2, max 16k tokens output
- **GPT-4-turbo**: Previous gen, supports temp 0-2, max 4k tokens output
- **GPT-4o-mini**: Budget option, same limits as GPT-4o

### Reasoning Models
- **O1-preview**: Best reasoning, temp always 1.0, max 128k thinking tokens
- **O1-mini**: Faster reasoning, temp always 1.0, max 65k thinking tokens

## Results Analysis

### LangSmith Dashboard
View results at: https://smith.langchain.com
- Filter by experiment name
- Compare metrics across experiments
- View individual run traces

### Local Results
```
evaluation_results/
├── eval_report_*.json          # Detailed results
├── eval_summary_*.md           # Human-readable summary

comparison_results/
├── model_comparison_*.json     # Comparison data
├── model_comparison_report_*.md # Analysis report
└── plots/                      # Visualization charts
    ├── model_comparison_*.png
    ├── temperature_impact_*.png
    └── agent_heatmap_*.png
```

## Recommended Evaluation Strategy

### 1. Establish Baseline
```bash
python run_quick_eval.py
# Select option 1 (Baseline)
```

### 2. Model Selection
```bash
python run_model_comparison.py
# Select option 2 (Model comparison)
```

### 3. Optimize Parameters
```bash
# For best model from step 2
python run_model_comparison.py
# Select option 3 (Temperature study)
# Select option 4 (Token study)
```

### 4. Cost-Quality Analysis
Review `comparison_results/*_report_*.md` for:
- Quality/Cost ratios
- Best configuration for your needs
- Agent-specific performance

## Interpreting Results

### Good Scores
- **Workflow Success**: >85% (6+ outputs generated)
- **End-to-End Quality**: >70%
- **Agent Quality**: >65% per agent

### Red Flags
- Any agent scoring <50%
- Workflow success <70% (missing outputs)
- Large variance between experiments

### Optimization Tips
- **Low Research scores**: Increase max tokens, lower temperature
- **Low Ideation scores**: Increase temperature (0.7-0.9)
- **Low Specification scores**: Use GPT-4o with 8k+ tokens
- **Missing outputs**: Check recursion limits, increase delays

## Cost Considerations

Estimated costs per full workflow run:
- **GPT-4o-mini**: ~$0.05-0.10 (most cost-effective)
- **GPT-4o**: ~$0.50-1.00 (best quality/cost)
- **GPT-4-turbo**: ~$1.00-1.50
- **O1-mini**: ~$0.30-0.60 (good for complex reasoning)
- **O1-preview**: ~$3.00-5.00 (best for hardest problems)

## Troubleshooting

### Missing Outputs
- Check `recursion_limit` in workflow config (set to 100+)
- Verify all agent tools are properly configured
- Review agent handoff logic in supervisor

### Low Scores
- Check if prompts match evaluation criteria
- Ensure sufficient context is passed between agents
- Verify file paths match expected names

### Evaluation Errors
- Ensure dataset exists: `python create_dataset.py`
- Check LangSmith API key is set
- Verify OpenAI API key has sufficient credits

## Next Steps

1. Run baseline evaluation
2. Identify weakest agent from per-agent scores
3. Optimize that agent's prompt/parameters
4. Re-run evaluation to measure improvement
5. Iterate until satisfied with scores

For questions or issues, check the generated reports and LangSmith traces for detailed debugging information.
