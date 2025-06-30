# R&D Workflow Evaluation System

## Overview

This evaluation system provides comprehensive quality metrics for your multi-agent R&D workflow, including per-agent performance scoring and overall workflow assessment.

## Evaluation Metrics

### Per-Agent Quality Scores (0-100%)

1. **Research Agent**: Market analysis depth, technical research quality, source diversity
2. **Ideation Agent**: Innovation level, concept diversity, feasibility assessment  
3. **Evaluation Agent**: Selection rationale, strategic analysis, implementation planning
4. **Specification Agent**: Technical completeness, architecture quality, documentation clarity
5. **Testing Agent**: Test coverage, validation methodology, quality metrics

### Overall Workflow Metrics

- **Workflow Success**: Counts completed outputs (0-7), each output = 1/7 of score
- **End-to-End Quality**: Overall request fulfillment and coherence assessment

## Quick Start

### 1. Create Evaluation Dataset (One-time Setup)

```bash
python create_dataset.py
```

This creates a persistent dataset with test cases that will be reused for all evaluations.

### 2. Run Quick Evaluations

```bash
python run_quick_eval.py
```

**Interactive Menu Options:**
1. **Run Baseline** (GPT-4o standard) - Quick performance baseline
2. **Quick Model Comparison** - Test multiple models (gpt-4o-mini, gpt-4.1, gpt-4.1-mini)
3. **Different Model Evaluation** - Test a specific model configuration
4. **Temperature Impact Test** - Compare temperature settings (0.2 vs 0.9)
5. **All Tests** - Run complete evaluation suite

### 3. Evaluation via Main Script

```bash
python main.py --evaluate
```

**How --evaluate Works:**

When you run `main.py --evaluate`, the system:

1. **Imports Evaluation Framework**: Loads the `AgentLabEvaluationSuite` from `agent_lab_evals.py`
2. **Runs Comprehensive Assessment**: Executes your workflow against the test dataset
3. **Evaluates Each Agent**: Scores individual agent performance using LLM-as-a-Judge methodology
4. **Assesses Overall Quality**: Evaluates end-to-end workflow effectiveness
5. **Generates Reports**: Creates detailed results in `evaluation_results/` directory

**Output Files:**
```
evaluation_results/
├── eval_report_*.json          # Detailed numerical results
└── eval_summary_*.md           # Human-readable summary with insights
```

## Understanding Results

### Good Performance Indicators
- **Workflow Success**: >85% (6+ outputs generated reliably)
- **End-to-End Quality**: >70% (coherent, professional outputs)
- **Agent Quality**: >65% per agent (effective specialized performance)

### Performance Issues
- Any agent scoring <50% (needs prompt/configuration improvement)
- Workflow success <70% (missing outputs, system errors)
- Large variance between runs (inconsistent performance)

## Configuration

### Model Selection
Set environment variables in `.env`:
```env
EXPERIMENT_LLM_MODEL=gpt-4o          # Model to evaluate
EXPERIMENT_TEMPERATURE=0.7           # Temperature setting
EXPERIMENT_MAX_TOKENS=8000          # Token limit
```

### Quick Model Testing
The quick evaluation tool tests these configurations:
- **gpt-4o-mini**: Most cost-effective (~$0.05-0.10 per run)
- **gpt-4.1**: Balanced performance (~$0.50-1.00 per run)  
- **gpt-4.1-mini**: Budget option with good performance

## Interpreting Results

### Example Output
```
📊 Per-Agent Quality Scores:
  Research Agent: 67%
  Ideation Agent: 74%  
  Evaluation Agent: 71%
  Specification Agent: 58%
  Testing Agent: 82%

🏆 Best Performing Agent: Testing (82%)
⚠️  Needs Improvement: Specification (58%)
📈 Workflow Completion Rate: 94% (6.6/7 outputs on average)
```

### Optimization Tips
- **Low Research scores**: Increase max tokens, improve search tools
- **Low Ideation scores**: Increase temperature for more creativity
- **Low Specification scores**: Use larger models, add technical knowledge
- **Missing outputs**: Check agent handoff logic, increase recursion limits


## Next Steps

1. **Run Baseline**: Start with `python run_quick_eval.py` → Option 1
2. **Identify Weakest Agent**: Look at per-agent scores in results
3. **Optimize Configuration**: Adjust model/temperature for weak agents
4. **Re-evaluate**: Run evaluation again to measure improvements
5. **Production Deploy**: Use optimal configuration for live workflows

For detailed debugging, check LangSmith traces at: https://smith.langchain.com

---

**Quick Commands Summary:**
```bash
# Setup (once)
python create_dataset.py

# Quick interactive evaluation  
python run_quick_eval.py

# Full evaluation via main script
python main.py --evaluate
```
