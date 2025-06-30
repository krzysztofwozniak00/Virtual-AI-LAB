"""
 LangSmith Evaluation Suite for R&D Multi-Agent Workflow
Includes individual agent quality evaluations and improved workflow metrics
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import uuid

from langsmith import Client, aevaluate
from langsmith.evaluation import EvaluationResult
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from loguru import logger

# Initialize LangSmith client
client = Client()

class EvaluationCase(BaseModel):
    """Model for evaluation test cases"""
    request: str
    workflow_type: str
    expected_outputs: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


#  evaluator functions for each agent
def evaluate_research_agent_quality(run, example) -> dict:
    """Evaluate the quality of research agent's output specifically"""
    outputs = run.outputs or {}
    research_brief = outputs.get("research_brief", "")
    
    if not research_brief:
        return {
            "key": "research_agent_quality",
            "score": 0,
            "comment": "No research brief generated"
        }
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Evaluate the research agent's output on these specific criteria:
        1. Market Analysis Depth (0-10): Comprehensive market size, trends, competitive landscape
        2. Technical Research Quality (0-10): Academic sources, technical feasibility, innovation landscape
        3. Source Diversity (0-10): Web search, market intelligence, academic papers utilized
        4. Strategic Insights (0-10): Actionable findings, opportunity identification
        5. Structure & Clarity (0-10): Well-organized, executive-ready presentation
        
        Respond with ONLY a number from 0-10 representing the overall research agent performance."""),
        ("human", "Request: {request}\n\nResearch Output:\n{research_brief}")
    ])
    
    try:
        response = llm.invoke(
            prompt.format_messages(
                request=example.inputs["request"],
                research_brief=research_brief[:3000]
            )
        )
        
        score_text = response.content.strip()
        score = float(score_text) if score_text.replace('.', '').isdigit() else 5.0
        
        return {
            "key": "research_agent_quality",
            "score": score / 10,
            "comment": f"Research agent score: {score}/10"
        }
    except Exception as e:
        return {
            "key": "research_agent_quality",
            "score": 0.5,
            "comment": f"Evaluation error: {str(e)}"
        }


def evaluate_ideation_agent_quality(run, example) -> dict:
    """Evaluate the quality of ideation agent's concept generation"""
    outputs = run.outputs or {}
    concepts = outputs.get("concepts", "")
    
    if not concepts:
        return {
            "key": "ideation_agent_quality",
            "score": 0,
            "comment": "No concepts generated"
        }
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Evaluate the ideation agent's output on these specific criteria:
        1. Innovation Level (0-10): Creativity, novelty, breakthrough thinking
        2. Concept Diversity (0-10): Range of different approaches and solutions
        3. Feasibility Assessment (0-10): Realistic evaluation of implementation
        4. Strategic Alignment (0-10): Alignment with research findings and market needs
        5. Concept Completeness (0-10): Detailed descriptions, value propositions, differentiation
        
        Respond with ONLY a number from 0-10 representing the overall ideation agent performance."""),
        ("human", "Request: {request}\n\nConcepts Output:\n{concepts}")
    ])
    
    try:
        response = llm.invoke(
            prompt.format_messages(
                request=example.inputs["request"],
                concepts=concepts[:3000]
            )
        )
        
        score_text = response.content.strip()
        score = float(score_text) if score_text.replace('.', '').isdigit() else 5.0
        
        return {
            "key": "ideation_agent_quality",
            "score": score / 10,
            "comment": f"Ideation agent score: {score}/10"
        }
    except Exception as e:
        return {
            "key": "ideation_agent_quality",
            "score": 0.5,
            "comment": f"Evaluation error: {str(e)}"
        }


def evaluate_evaluation_agent_quality(run, example) -> dict:
    """Evaluate the quality of evaluation/selection agent's analysis"""
    outputs = run.outputs or {}
    selected_concept = outputs.get("selected_concept", "")
    
    if not selected_concept:
        return {
            "key": "evaluation_agent_quality",
            "score": 0,
            "comment": "No concept selection/analysis generated"
        }
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Evaluate the evaluation agent's output on these specific criteria:
        1. Selection Rationale (0-10): Clear justification for concept selection
        2. Strategic Analysis Depth (0-10): Market fit, competitive positioning, risks
        3. Implementation Planning (0-10): Timeline, resources, partnerships identified
        4. Financial Considerations (0-10): Business model, investment requirements
        5. Risk Assessment Quality (0-10): Comprehensive risk analysis and mitigation
        
        Respond with ONLY a number from 0-10 representing the overall evaluation agent performance."""),
        ("human", "Request: {request}\n\nSelection Analysis:\n{selected_concept}")
    ])
    
    try:
        response = llm.invoke(
            prompt.format_messages(
                request=example.inputs["request"],
                selected_concept=selected_concept[:3000]
            )
        )
        
        score_text = response.content.strip()
        score = float(score_text) if score_text.replace('.', '').isdigit() else 5.0
        
        return {
            "key": "evaluation_agent_quality",
            "score": score / 10,
            "comment": f"Evaluation agent score: {score}/10"
        }
    except Exception as e:
        return {
            "key": "evaluation_agent_quality",
            "score": 0.5,
            "comment": f"Evaluation error: {str(e)}"
        }


def evaluate_specification_agent_quality(run, example) -> dict:
    """Evaluate the quality of specification agent's technical specs"""
    outputs = run.outputs or {}
    specifications = outputs.get("specifications", "")
    
    if not specifications:
        return {
            "key": "specification_agent_quality",
            "score": 0,
            "comment": "No specifications generated"
        }
    
    # Check for key technical elements with more granularity
    key_elements = [
        "functional requirements", "non-functional requirements", "system architecture",
        "components", "interfaces", "data models", "api specifications",
        "performance metrics", "scalability considerations", "security requirements",
        "integration points", "technology stack", "deployment architecture"
    ]
    
    specs_lower = specifications.lower()
    found_elements = sum(1 for elem in key_elements if elem in specs_lower)
    completeness_score = found_elements / len(key_elements)
    
    # Also evaluate with LLM for quality
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Evaluate the specification agent's output on these criteria:
        1. Technical Completeness (0-10): All necessary specs covered
        2. Implementation Detail (0-10): Sufficient detail for development
        3. Architecture Quality (0-10): Well-designed system architecture
        4. Standards Compliance (0-10): Following industry best practices
        5. Clarity & Documentation (0-10): Clear, unambiguous specifications
        
        Respond with ONLY a number from 0-10 representing the overall specification quality."""),
        ("human", "Specifications:\n{specifications}")
    ])
    
    try:
        response = llm.invoke(
            prompt.format_messages(
                specifications=specifications[:3000]
            )
        )
        
        score_text = response.content.strip()
        llm_score = float(score_text) if score_text.replace('.', '').isdigit() else 5.0
        
        # Combine completeness check and LLM evaluation
        final_score = (completeness_score + (llm_score / 10)) / 2
        
        return {
            "key": "specification_agent_quality",
            "score": final_score,
            "comment": f"Specification agent score: {final_score:.2f} (found {found_elements}/{len(key_elements)} key elements)"
        }
    except Exception as e:
        return {
            "key": "specification_agent_quality",
            "score": completeness_score,
            "comment": f"Partial evaluation: {found_elements}/{len(key_elements)} elements found"
        }


def evaluate_testing_agent_quality(run, example) -> dict:
    """Evaluate the quality of testing agent's validation plan"""
    outputs = run.outputs or {}
    validation_plan = outputs.get("validation_plan", "")
    
    if not validation_plan:
        return {
            "key": "testing_agent_quality",
            "score": 0,
            "comment": "No testing/validation plan generated"
        }
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Evaluate the testing agent's output on these specific criteria:
        1. Test Coverage (0-10): Unit, integration, system, acceptance testing
        2. Validation Methodology (0-10): Clear test strategies and approaches
        3. Quality Metrics (0-10): KPIs, success criteria, measurement framework
        4. Risk-Based Testing (0-10): Focus on critical areas and edge cases
        5. Compliance & Standards (0-10): Regulatory requirements, industry standards
        
        Respond with ONLY a number from 0-10 representing the overall testing agent performance."""),
        ("human", "Validation Plan:\n{validation_plan}")
    ])
    
    try:
        response = llm.invoke(
            prompt.format_messages(
                validation_plan=validation_plan[:3000]
            )
        )
        
        score_text = response.content.strip()
        score = float(score_text) if score_text.replace('.', '').isdigit() else 5.0
        
        return {
            "key": "testing_agent_quality",
            "score": score / 10,
            "comment": f"Testing agent score: {score}/10"
        }
    except Exception as e:
        return {
            "key": "testing_agent_quality",
            "score": 0.5,
            "comment": f"Evaluation error: {str(e)}"
        }


def evaluate_workflow_success_v2(run, example) -> dict:
    """ workflow success based on output count (0-7)"""
    outputs = run.outputs or {}
    
    # All possible outputs in order
    expected_outputs = [
        "research_brief",      # 1. Research agent
        "concepts",           # 2. Ideation agent
        "selected_concept",   # 3. Evaluation agent
        "specifications",     # 4. Specification agent
        "visualization",      # 5. Visualization agent (image)
        "validation_plan",    # 6. Testing agent
        "final_report"        # 7. Documentation agent
    ]
    
    # Count generated outputs
    generated_count = sum(1 for output in expected_outputs if outputs.get(output))
    
    # Calculate score (0-7 outputs = 0-1 score)
    score = generated_count / len(expected_outputs)
    
    # Check for errors
    error_msg = ""
    if outputs.get("success") is False:
        error_msg = f" | Error: {outputs.get('error', 'Unknown error')}"
    
    comment = f"Generated {generated_count}/{len(expected_outputs)} outputs{error_msg}"
    
    # List missing outputs if any
    if generated_count < len(expected_outputs):
        missing = [out for out in expected_outputs if not outputs.get(out)]
        comment += f" | Missing: {', '.join(missing)}"
    
    return {
        "key": "workflow_success_v2",
        "score": score,
        "comment": comment
    }


def evaluate_end_to_end_quality(run, example) -> dict:
    """Evaluate overall end-to-end quality of the workflow"""
    outputs = run.outputs or {}
    request = example.inputs.get("request", "")
    
    # Combine key outputs for overall evaluation
    combined_output = ""
    key_outputs = ["research_brief", "concepts", "selected_concept", "specifications", "validation_plan"]
    
    for key in key_outputs:
        if outputs.get(key):
            combined_output += f"\n\n=== {key.upper()} ===\n{outputs[key][:1000]}"
    
    if not combined_output:
        return {
            "key": "end_to_end_quality",
            "score": 0,
            "comment": "No outputs to evaluate end-to-end quality"
        }
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Evaluate the overall end-to-end workflow quality:
        1. Request Fulfillment (0-10): How well does the complete workflow address the original request?
        2. Coherence (0-10): Do all agent outputs flow logically and build upon each other?
        3. Professional Quality (0-10): Executive-ready outputs, proper formatting, clarity
        4. Innovation to Implementation (0-10): Clear path from idea to executable plan
        5. Strategic Value (0-10): Actionable insights and recommendations provided
        
        Respond with ONLY a number from 0-10 representing overall workflow quality."""),
        ("human", "Original Request: {request}\n\nWorkflow Outputs:\n{outputs}")
    ])
    
    try:
        response = llm.invoke(
            prompt.format_messages(
                request=request,
                outputs=combined_output
            )
        )
        
        score_text = response.content.strip()
        score = float(score_text) if score_text.replace('.', '').isdigit() else 5.0
        
        return {
            "key": "end_to_end_quality",
            "score": score / 10,
            "comment": f"End-to-end quality score: {score}/10"
        }
    except Exception as e:
        return {
            "key": "end_to_end_quality",
            "score": 0.5,
            "comment": f"Evaluation error: {str(e)}"
        }


class AgentLabEvaluationSuite:
    """ evaluation suite with per-agent quality metrics"""
    
    def __init__(self, dataset_name=None):
        self.client = client
        if dataset_name:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = f"rd_workflow_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def create_evaluation_dataset(self) -> str:
        """Create a dataset with test cases for evaluation"""
        
        test_cases = [
            # New Product Development Cases
            EvaluationCase(
                request="Develop an AI-powered personal health monitoring device that can predict health issues before symptoms appear",
                workflow_type="new_product",
                tags=["healthcare", "ai", "iot", "new_product"],
                expected_outputs={
                    "should_include_market_research": "healthcare wearables market analysis",
                    "should_include_ai_tech": "predictive analytics, machine learning models",
                    "should_include_sensors": "biosensors, vital sign monitoring"
                }
            ),
            
            EvaluationCase(
                request="Create a sustainable packaging solution for e-commerce that reduces waste by 80%",
                workflow_type="new_product",
                tags=["sustainability", "packaging", "e-commerce", "new_product"],
                expected_outputs={
                    "should_include_materials": "biodegradable, recyclable materials",
                    "should_include_metrics": "waste reduction metrics, lifecycle analysis"
                }
            ),
            
            # Complex Innovation Cases
            EvaluationCase(
                request="Design a quantum computing solution for drug discovery that reduces development time by 10x",
                workflow_type="new_product",
                tags=["quantum", "pharma", "complex", "new_product"],
                expected_outputs={
                    "should_include_quantum": "quantum algorithms, qubit requirements",
                    "should_include_pharma": "molecular simulation, drug targets"
                }
            )
        ]
        
        # Create dataset in LangSmith
        dataset = self.client.create_dataset(
            dataset_name=self.dataset_name,
            description=" evaluation dataset for R&D multi-agent workflow with per-agent metrics"
        )
        
        # Add examples to dataset
        for case in test_cases:
            self.client.create_example(
                inputs={
                    "request": case.request,
                    "workflow_type": case.workflow_type
                },
                outputs=case.expected_outputs,
                dataset_id=dataset.id,
                metadata={"tags": case.tags}
            )
        
        logger.info(f"Created dataset '{self.dataset_name}' with {len(test_cases)} examples")
        return dataset.id
    
    def get_evaluators(self) -> List[Callable]:
        """Get all evaluators including per-agent quality metrics"""
        return [
            # Per-agent quality evaluators
            evaluate_research_agent_quality,
            evaluate_ideation_agent_quality,
            evaluate_evaluation_agent_quality,
            evaluate_specification_agent_quality,
            evaluate_testing_agent_quality,
            
            # Overall workflow evaluators
            evaluate_workflow_success_v2,
            evaluate_end_to_end_quality,
        ]
    
    async def run_evaluation(
        self, 
        workflow_function: Callable,
        dataset_id: Optional[str] = None,
        experiment_prefix: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Run evaluation of the R&D workflow"""
        
        # Use existing dataset if no dataset_id provided
        if not dataset_id:
            try:
                datasets = list(self.client.list_datasets(dataset_name=self.dataset_name))
                if datasets:
                    dataset_id = datasets[0].id
                    logger.info(f"Using existing dataset: {self.dataset_name}")
                else:
                    dataset_id = self.create_evaluation_dataset()
            except:
                dataset_id = self.create_evaluation_dataset()
        
        # Create unique experiment prefix if not provided
        if not experiment_prefix:
            experiment_prefix = f"rd_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Merge default metadata with provided metadata
        default_metadata = {
            "evaluator_version": "2.0",  # Updated version
            "workflow_type": "multi_agent_rd",
            "timestamp": datetime.now().isoformat(),
            "includes_per_agent_metrics": True
        }
        if metadata:
            default_metadata.update(metadata)
        
        logger.info(f"Starting  evaluation with experiment prefix: {experiment_prefix}")
        
        try:
            # Run the evaluation
            results = await aevaluate(
                workflow_function,
                data=self.dataset_name,
                evaluators=self.get_evaluators(),
                experiment_prefix=experiment_prefix,
                metadata=default_metadata,
                description=" R&D workflow evaluation with per-agent quality metrics",
                max_concurrency=2,
                client=self.client
            )
            
            # Process results
            summary = self._process_results(results)
            summary["experiment_name"] = experiment_prefix
            
            # Save evaluation report
            self._save_evaluation_report(summary)
            
            logger.info(" evaluation completed successfully!")
            return summary
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def _process_results(self, results) -> Dict[str, Any]:
        """Process evaluation results with agent-specific breakdowns"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset": self.dataset_name,
            "results": {}
        }
        
        # Extract aggregate metrics if available
        if hasattr(results, 'aggregate_metrics'):
            summary["aggregate_metrics"] = results.aggregate_metrics
        
        # Process individual results
        if hasattr(results, 'results'):
            summary["total_runs"] = len(results.results)
            summary["evaluator_scores"] = {}
            summary["agent_scores"] = {
                "research": [],
                "ideation": [],
                "evaluation": [],
                "specification": [],
                "testing": []
            }
            
            # Calculate average scores per evaluator
            for result in results.results:
                if hasattr(result, 'evaluation_results'):
                    for eval_name, eval_result in result.evaluation_results.items():
                        if eval_name not in summary["evaluator_scores"]:
                            summary["evaluator_scores"][eval_name] = []
                        if isinstance(eval_result, dict) and "score" in eval_result:
                            summary["evaluator_scores"][eval_name].append(eval_result["score"])
                            
                            # Track agent-specific scores
                            if "research_agent_quality" in eval_name:
                                summary["agent_scores"]["research"].append(eval_result["score"])
                            elif "ideation_agent_quality" in eval_name:
                                summary["agent_scores"]["ideation"].append(eval_result["score"])
                            elif "evaluation_agent_quality" in eval_name:
                                summary["agent_scores"]["evaluation"].append(eval_result["score"])
                            elif "specification_agent_quality" in eval_name:
                                summary["agent_scores"]["specification"].append(eval_result["score"])
                            elif "testing_agent_quality" in eval_name:
                                summary["agent_scores"]["testing"].append(eval_result["score"])
            
            # Calculate averages
            summary["average_scores"] = {}
            for eval_name, scores in summary["evaluator_scores"].items():
                if scores:
                    summary["average_scores"][eval_name] = sum(scores) / len(scores)
            
            # Calculate per-agent averages
            summary["average_agent_scores"] = {}
            for agent, scores in summary["agent_scores"].items():
                if scores:
                    summary["average_agent_scores"][agent] = sum(scores) / len(scores)
        
        return summary
    
    def _save_evaluation_report(self, summary: Dict[str, Any]):
        """Save  evaluation report with agent breakdowns"""
        Path("evaluation_results").mkdir(exist_ok=True)
        
        report_path = f"evaluation_results/eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        # Create  human-readable summary
        summary_path = f"evaluation_results/eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(summary_path, "w") as f:
            f.write("#  R&D Workflow Evaluation Summary\n\n")
            f.write(f"**Date**: {summary['timestamp']}\n")
            f.write(f"**Dataset**: {summary['dataset']}\n\n")
            
            if "average_scores" in summary:
                f.write("## Overall Metrics\n\n")
                for metric, score in summary["average_scores"].items():
                    f.write(f"- **{metric}**: {score:.2%}\n")
            
            if "average_agent_scores" in summary:
                f.write("\n## Per-Agent Quality Scores\n\n")
                for agent, score in summary["average_agent_scores"].items():
                    f.write(f"- **{agent.capitalize()} Agent**: {score:.2%}\n")
            
            # Add performance insights
            if "average_agent_scores" in summary and summary["average_agent_scores"]:
                f.write("\n## Performance Insights\n\n")
                
                # Find best and worst performing agents
                agent_scores = summary["average_agent_scores"]
                if agent_scores:
                    best_agent = max(agent_scores, key=agent_scores.get)
                    worst_agent = min(agent_scores, key=agent_scores.get)
                    
                    f.write(f"- **Best Performing Agent**: {best_agent.capitalize()} ({agent_scores[best_agent]:.2%})\n")
                    f.write(f"- **Needs Improvement**: {worst_agent.capitalize()} ({agent_scores[worst_agent]:.2%})\n")
                
                # Workflow success rate
                if "workflow_success_v2" in summary.get("average_scores", {}):
                    success_rate = summary["average_scores"]["workflow_success_v2"]
                    f.write(f"- **Workflow Completion Rate**: {success_rate:.2%} ({success_rate * 7:.1f}/7 outputs on average)\n")
            
            f.write("\n## Detailed Results\n")
            f.write(f"\nFull results available in: {report_path}\n")
        
        logger.info(f" evaluation summary saved to {summary_path}")
    
    async def run_single_evaluation(
        self,
        workflow_function: Callable,
        test_request: str,
        workflow_type: str = "new_product"
    ) -> Dict[str, Any]:
        """Run evaluation on a single test case"""
        
        logger.info(f"Running single evaluation for: {test_request}")
        
        # Create temporary dataset with single example
        dataset = self.client.create_dataset(
            dataset_name=f"single_eval_{uuid.uuid4().hex[:8]}",
            description="Single test case evaluation"
        )
        
        self.client.create_example(
            inputs={
                "request": test_request,
                "workflow_type": workflow_type
            },
            outputs={},
            dataset_id=dataset.id
        )
        
        # Run evaluation
        results = await self.run_evaluation(
            workflow_function,
            dataset_id=dataset.id
        )
        
        # Clean up temporary dataset
        try:
            self.client.delete_dataset(dataset_id=dataset.id)
        except:
            pass
        
        return results


# Standalone evaluation functions
async def evaluate_rd_workflow(workflow_function: Callable) -> Dict[str, Any]:
    """Simple function to evaluate R&D workflow"""
    suite = AgentLabEvaluationSuite()
    return await suite.run_evaluation(workflow_function)


async def quick_test_workflow(workflow_function: Callable, request: str) -> Dict[str, Any]:
    """Quick test a single request"""
    suite = AgentLabEvaluationSuite()
    return await suite.run_single_evaluation(workflow_function, request)


if __name__ == "__main__":
    print(" R&D Workflow Evaluation Suite")
   