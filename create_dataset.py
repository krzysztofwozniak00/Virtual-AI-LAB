from agent_lab_evals import AgentLabEvaluationSuite
from langsmith import Client

def create_persistent_dataset():
    """Create a dataset that will be reused for all experiments"""
    suite = AgentLabEvaluationSuite()
    
    # Override the dataset name to make it persistent
    suite.dataset_name = "rd_workflow_eval_persistent"  # Fixed name instead of timestamp
    
    # Create the dataset
    dataset_id = suite.create_evaluation_dataset()
    
    print(f"✅ Created persistent dataset: {suite.dataset_name}")
    print(f"📝 Dataset ID: {dataset_id}")
    print(f"🔗 View in LangSmith: https://smith.langchain.com/datasets/{dataset_id}")
    
    return suite.dataset_name, dataset_id

if __name__ == "__main__":
    create_persistent_dataset()