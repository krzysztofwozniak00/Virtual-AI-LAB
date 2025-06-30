#!/usr/bin/env python3
"""
Quick evaluation commands for R&D workflow model comparison
"""

import asyncio
import os
import sys
from datetime import datetime
from agent_lab_evals import AgentLabEvaluationSuite
from main import run_workflow_for_evaluation

async def quick_baseline():
    """Run baseline evaluation with GPT-4o"""
    print("🚀 Running Baseline Evaluation (GPT-4o)")
    
    os.environ["EXPERIMENT_LLM_MODEL"] = "gpt-4o"
    os.environ["EXPERIMENT_TEMPERATURE"] = "0.7"
    os.environ["EXPERIMENT_MAX_TOKENS"] = "8000"
    
    suite = AgentLabEvaluationSuite(dataset_name="rd_workflow_eval_persistent")
    
    results = await suite.run_evaluation(
        workflow_function=run_workflow_for_evaluation,
        experiment_prefix=f"baseline_gpt4o_8k_0.7__improved_prompt{datetime.now().strftime('%Y%m%d_%H%M')}",
        metadata={
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 8000,
            "experiment_type": "baseline"
        }
    )
    
    print("\n📊 Baseline Results:")
    print(f"Workflow Success: {results.get('average_scores', {}).get('workflow_success_v2', 0):.2%}")
    print(f"End-to-End Quality: {results.get('average_scores', {}).get('end_to_end_quality', 0):.2%}")
    
    print("\nPer-Agent Scores:")
    for agent, score in results.get('average_agent_scores', {}).items():
        print(f"  {agent.capitalize()}: {score:.2%}")

async def quick_diffrent_model_evaluation():
    """Run baseline evaluation with different model"""
    print("🚀 Running Evaluation for:")
    
    os.environ["EXPERIMENT_LLM_MODEL"] = "gpt-4.1"
    os.environ["EXPERIMENT_TEMPERATURE"] = "0.7"
    os.environ["EXPERIMENT_MAX_TOKENS"] = "16000"
    
    suite = AgentLabEvaluationSuite(dataset_name="rd_workflow_eval_persistent")
    
    results = await suite.run_evaluation(
        workflow_function=run_workflow_for_evaluation,
        experiment_prefix=f"gpt4.1_16k_{datetime.now().strftime('%Y%m%d_%H%M')}",
        metadata={
            "model": "gpt-4.1",
            "temperature": 0.7,
            "max_tokens": 16000,
            "experiment_type": "diffrent_model"
        }
    )
    
    print("\n📊 Baseline Results:")
    print(f"Workflow Success: {results.get('average_scores', {}).get('workflow_success_v2', 0):.2%}")
    print(f"End-to-End Quality: {results.get('average_scores', {}).get('end_to_end_quality', 0):.2%}")
    
    print("\nPer-Agent Scores:")
    for agent, score in results.get('average_agent_scores', {}).items():
        print(f"  {agent.capitalize()}: {score:.2%}")


async def quick_model_comparison():
    """Quick comparison of main models"""
    print("🔬 Running Quick Model Comparison")
    
    models = ["gpt-4o-mini","gpt-4.1","gpt-4.1-mini"]
    suite = AgentLabEvaluationSuite(dataset_name="rd_workflow_eval_persistent")
    
    results_summary = []
    
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Testing {model}...")
        
        # Handle o1 models differently
        temp = 1.0 if model.startswith("o1") else 0.7
        
        os.environ["EXPERIMENT_LLM_MODEL"] = model
        os.environ["EXPERIMENT_TEMPERATURE"] = str(temp)
        os.environ["EXPERIMENT_MAX_TOKENS"] = "8000"
        
        try:
            results = await suite.run_evaluation(
                workflow_function=run_workflow_for_evaluation,
                experiment_prefix=f"{model}_8k_{datetime.now().strftime('%Y%m%d_%H%M')}",
                metadata={
                    "model": model,
                    "temperature": temp,
                    "max_tokens": 8000,
                    "experiment_type": "diffrent_models_8k"
                }
            )
            
            results_summary.append({
                "model": model,
                "workflow_success": results.get('average_scores', {}).get('workflow_success_v2', 0),
                "quality": results.get('average_scores', {}).get('end_to_end_quality', 0),
                "avg_agent_score": sum(results.get('average_agent_scores', {}).values()) / len(results.get('average_agent_scores', {})) if results.get('average_agent_scores') else 0
            })
            
        except Exception as e:
            print(f"❌ Error with {model}: {e}")
        
        if i < len(models):
            print("⏳ Waiting 10s...")
            await asyncio.sleep(10)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<15} {'Success':<12} {'Quality':<12} {'Avg Agent':<12}")
    print("-"*51)
    
    for r in results_summary:
        print(f"{r['model']:<15} {r['workflow_success']:<12.2%} {r['quality']:<12.2%} {r['avg_agent_score']:<12.2%}")
    
    # Find best
    if results_summary:
        best_quality = max(results_summary, key=lambda x: x['quality'])
        best_success = max(results_summary, key=lambda x: x['workflow_success'])
        
        print(f"\n🏆 Best Quality: {best_quality['model']} ({best_quality['quality']:.2%})")
        print(f"🏆 Best Success: {best_success['model']} ({best_success['workflow_success']:.2%})")

async def quick_temperature_test():
    """Test temperature impact on GPT-4o"""
    print("🌡️ Running Temperature Impact Test (GPT-4o)")
    
    temperatures = [0.2, 0.9]
    suite = AgentLabEvaluationSuite(dataset_name="rd_workflow_eval_persistent")
    
    results_by_temp = {}
    
    for i, temp in enumerate(temperatures, 1):
        print(f"\n[{i}/{len(temperatures)}] Testing temperature {temp}...")
        
        os.environ["EXPERIMENT_LLM_MODEL"] = "gpt-4o"
        os.environ["EXPERIMENT_TEMPERATURE"] = str(temp)
        os.environ["EXPERIMENT_MAX_TOKENS"] = "8000"
        
        results = await suite.run_evaluation(
            workflow_function=run_workflow_for_evaluation,
            experiment_prefix=f"gpt_4o_8k_temp_{temp}_{datetime.now().strftime('%Y%m%d_%H%M')}",
            metadata={
                "model": "gpt-4o",
                "temperature": temp,
                "max_tokens": 8000,
                "experiment_type": "temperature_test"
            }
        )
        
        results_by_temp[temp] = {
            "ideation": results.get('average_agent_scores', {}).get('ideation', 0),
            "research": results.get('average_agent_scores', {}).get('research', 0),
            "specification": results.get('average_agent_scores', {}).get('specification', 0)
        }
        
        if i < len(temperatures):
            print("⏳ Waiting 10s...")
            await asyncio.sleep(10)
    
    # Show impact
    print("\n🌡️ TEMPERATURE IMPACT ON AGENTS")
    print("="*50)
    print(f"\n{'Temp':<8} {'Ideation':<12} {'Research':<12} {'Specification':<12}")
    print("-"*44)
    
    for temp, scores in results_by_temp.items():
        print(f"{temp:<8} {scores['ideation']:<12.2%} {scores['research']:<12.2%} {scores['specification']:<12.2%}")

async def main():
    """Main menu for quick evaluations"""
    
    print("🚀 R&D Workflow Quick Evaluation Tool")
    print("\nOptions:")
    print("1. Run Baseline (GPT-4o standard)")
    print("2. Quick Model Comparison")
    print("3. Quick Different Model Evaluation")
    print("4. Temperature Impact Test")
    print("5. All Tests")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        await quick_baseline()
    elif choice == "2":
        await quick_model_comparison()
    elif choice == "3":
        await quick_diffrent_model_evaluation()
    elif choice == "4":
        await quick_temperature_test()
    elif choice == "5":
        print("Running all tests...\n")
        await quick_baseline()
        print("\n" + "="*70 + "\n")
        await quick_model_comparison()
        print("\n" + "="*70 + "\n")
        await quick_temperature_test()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    asyncio.run(main())