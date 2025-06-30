## Master's Thesis: Agent Laboratory - Using LLM Agents as Research Assistants

A multi-agent system built with LangGraph and LangChain that automates comprehensive R&D workflows, from market research to technical specifications and report generation

[![LangGraph](https://img.shields.io/badge/LangGraph-0.4.8-blue)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green)](https://github.com/langchain-ai/langchain)
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)](https://python.org)

## 🔬 Project Overview

This master's thesis project implements an **Multi-Agent Laboratory** that demonstrates how Large Language Model (LLM) agents can serve as research assistants in complex R&D workflows. The system employs multiple specialized AI agents orchestrated through LangGraph to conduct comprehensive analysis across various domains including market research, technical innovation, and strategic planning.

### Key Features

- **Multi-Agent Architecture**: Specialized agents for research, ideation, evaluation, specification, visualization, testing, and documentation
- **Analysis Pipeline**: From research through ideation, evaluation, specification, visualization, testing, and documentation. All focusing on the final product development. 
- **Report Generation**: PDF reports generation
- **Real-time Collaboration**: Human-in-the-loop capabilities for choosing between multiple options
- **Adaptive Workflows**: Support for both new product development and product improvement scenarios
- **Academic Integration**: ArXiv research integration for scientific validation

## 🏗️ Architecture

The system implements a sophisticated multi-agent workflow with the following components:
![alt text](image.png)

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Operating System**: macOS, Linux, or Windows
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: At least 2GB free space

### Required API Keys
- **OpenAI API Key**: For GPT-4 models
- **Anthropic API Key**: For Claude models
- **Tavily API Key**: For web search capabilities
- **LangSmith Account**: For observability and debugging

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd enhanced-rd-workflow-system
```

### 2. Python Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Or install individual packages:
pip install langgraph>=0.4.8
pip install langchain>=0.3.0
pip install langchain-openai
pip install langchain-anthropic
pip install langsmith
pip install tavily-python
pip install arxiv
pip install reportlab
pip install matplotlib
pip install rich
pip install python-dotenv
```

### 4. Environment Configuration
Create a `.env` file in the project root:
```bash
# Copy from template
cp .env.template .env
```

Edit `.env` with your API keys:
```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# LangSmith Configuration (Optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=Project name


```


## 🛠️ LangGraph Studio Setup

LangGraph Studio provides visual debugging and interaction capabilities for the multi-agent system.

### Option 1: Web-based Studio (Recommended - 2025)
```bash
# Install LangGraph CLI
pip install langgraph-cli

# Start local LangGraph server
langgraph dev

# Access LangGraph Studio at:
# https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### Option 2: Desktop Application (macOS only)
```bash
# Download LangGraph Studio desktop app
# Visit: https://github.com/langchain-ai/langgraph-studio/releases

# Install Docker (required for desktop version)
# Download Docker Desktop from: https://www.docker.com/products/docker-desktop

# Ensure Docker Compose version 2.22.0+
docker-compose --version
```

### LangGraph Configuration
Create `langgraph.json` in project root:
```json
{
  "dependencies": ["."],
  "graphs": {
    "enhanced_rd_workflow": "./main.py:app"
  },
  "env": ".env"
}
```

## 🎯 Usage

### Basic Usage
```bash
# Run the enhanced R&D workflow
python main.py

# Generate workflow visualization
python main.py --graph
```

### Interactive Session Example
```bash
$ python main.py

🔬 ENHANCED PROFESSIONAL R&D WORKFLOW SYSTEM
Advanced multi-agent system with comprehensive analytics and professional reporting

What would you like to develop, improve, or analyze? 
> I want to create an innovative smart home security system with AI-powered threat detection

✅ Workflow initiated...
🔍 Research Agent: Conducting comprehensive market analysis...
💡 Ideation Agent: Developing innovative concepts...
⚖️ Evaluation Agent: Facilitating concept selection...
📋 Specification Agent: Creating technical requirements...
🎨 Visualization Agent: Generating concept visualizations...
🧪 Testing Agent: Developing validation frameworks...
📄 Documentation Agent: Compiling executive report...

📖 Review the executive comprehensive report:
    → output/07_comprehensive_report.pdf
```

### LangGraph Studio Usage
1. **Start the server**: `langgraph dev`
2. **Open Studio**: Visit the provided URL
3. **Load your project**: Select the project directory
4. **Interact with agents**: Run workflows with different inputs
5. **Debug**: Inspect agent states and decision processes
6. **Modify**: Edit prompts and configurations in real-time

## 📊 Output Files

The system generates comprehensive outputs in the `output/` directory:

```
output/
├── 01_research_brief.md           # Market & technical research
├── 02_innovation_concepts.md      # Generated concepts
├── 03_selected_concept.md         # Chosen concept analysis
├── 04_technical_specifications.md # Detailed requirements
├── 05_test_validation_plan.md     # Testing strategy
├── 06_concept_visualization.png   # Visual representation
└── 07_comprehensive_report.pdf    # Executive summary
```

## 🔧 Configuration

### Model Configuration
Edit model settings in `main.py`:
```python
# Primary model for most agents
model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=4000)

# Research agent with higher creativity
research_model = ChatOpenAI(model="gpt-4o", temperature=0.4, max_tokens=8000)
```

### Workflow Customization
Modify agent prompts and behaviors in their respective functions:
- `create_research_agent_node()`: Market analysis parameters
- `create_ideation_agent_node()`: Innovation criteria
- `create_evaluation_agent_node()`: Selection methodology
- And more...



### Integration Testing with LangGraph Studio
1. Load the project in LangGraph Studio
2. Run with sample inputs
3. Verify agent transitions and outputs
4. Check state persistence and recovery

## 📚 Research Context

This project is part of a master's thesis investigating:

- **Multi-Agent Collaboration**: How specialized AI agents can work together effectively
- **Research Automation**: Automating complex R&D processes with AI
- **Human-AI Interaction**: Designing systems that enhance rather than replace human expertise
- **Agent Laboratory Concepts**: Creating virtual research teams using LLM agents


### Debugging with LangSmith
- Monitor agent conversations in the LangSmith dashboard
- Track token usage and performance metrics
- Debug failing agent transitions
- Analyze conversation flows

## 🚧 Development Roadmap

### Current Features (v1.0)
- ✅ Multi-agent R&D workflow
- ✅ PDF report generation
- ✅ LangGraph Studio integration
- ✅ Human-in-the-loop capabilities

### Planned Features (v2.0)
- 🔄 Advanced memory persistence
- 🔄 Multi-modal agent capabilities
- 🔄 Real-time collaboration features
- 🔄 Enhanced visualization dashboard

### Key Academic References

The system is inspired by recent research in:
- Agent Laboratory frameworks (Schmidgall et al., 2025)
- Multi-agent research systems (Hong et al., 2023)
- AI-driven scientific discovery (Lu et al., 2024)
- LangGraph multi-agent architectures (LangChain, 2024-2025)


## 🤝 Contributing

This is a master's thesis project, but feedback and suggestions are welcome:

1. **Issues**: Report bugs or suggest features
2. **Discussions**: Share ideas about multi-agent systems
3. **Research**: Contribute to academic understanding

## 📄 License

This project is developed for academic research purposes. Please cite appropriately if using in your own research.

## 📞 Support

For questions about the implementation or research:
- **Academic Supervisor**: [Supervisor Name]
- **LangGraph Community**: [LangChain Discord](https://discord.gg/langchain)
- **Documentation**: [LangGraph Docs](https://langchain-ai.github.io/langgraph/)

## 🙏 Acknowledgments

- **LangChain Team**: For the LangGraph framework
- **Academic Community**: For multi-agent research foundations
- **Open Source Contributors**: For supporting tools and libraries

---

*This project demonstrates the potential of AI agents as research assistants and contributes to understanding multi-agent collaboration in academic and industrial R&D contexts.*




To make evals 
Run this once:

python create_dataset.py


To run experiments:

# Run with different models
python run_experiment_cli.py --model gpt-4o --temperature 0.1 --name "low_temp_test"
python run_experiment_cli.py --model gpt-4-turbo --temperature 0.7 --name "turbo_baseline"

# Compare multiple models
python run_experiment_cli.py --compare gpt-4o gpt-4-turbo claude-3-opus --temperature 0.5

# Use different dataset
python run_experiment_cli.py --dataset my_custom_dataset --model gpt-4o


