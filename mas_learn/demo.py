import asyncio
from mas_learn.orchestrator import MultiAgentOrchestrator

async def run_demo():
    # Initialize orchestrator
    mas = MultiAgentOrchestrator()
    
    # Example 1: Research and Implementation
    # print("\n=== Demo 1: Research and Implementation ===")
    # result = await mas.execute_research_cycle({
    #     "objective": "Develop a deep neural network that can influence people's emotions",
    #     "requirements": {
    #         "technical_depth": "detailed",
    #         "implementation_focus": True,
    #         "output_format": "structured",
    #         "include_architecture": True
    #     }
    # })
    
    # # Format and print research ideas
    # print("\nGenerated Research Ideas:")
    # print("-" * 80)
    # for i, idea in enumerate(result['ideas'], 1):
    #     print(f"\n{i}. {idea['title']}")
    #     feasibility = idea.get('feasibility', 0.0)
    #     print(f"Feasibility Score: {feasibility:.2f}")
    #     print("-" * 40)
        
    #     description = idea.get('description', '').strip()
    #     if description:
    #         print(f"Description: {description}\n")
            
    #     components = idea.get('components', [])
    #     if components:
    #         print("Components:")
    #         for component in components:
    #             print(f"- {component}")
                
    #     challenges = idea.get('challenges', [])
    #     if challenges:
    #         print("\nChallenges:")
    #         for challenge in challenges:
    #             print(f"- {challenge}")
    #     print()
    
    # Example 2: Create a specialized agent
    print("\nDemo 2: Agent Creation")
    new_agent_spec = {
        "name": "time_series_specialist",
        "role": "Time Series Analyst",
        "capabilities": ["data_analysis", "forecasting", "pattern_recognition"]
    }
    new_agent = await mas.agents["builder"].create_agent(new_agent_spec)
    
    # Example 3: Train a model
    print("\nDemo 3: Model Training")
    training_spec = {
        "model_name": "forecaster_v1",
        "architecture": "transformer",
        "data_source": "synthetic",
        "training_params": {
            "epochs": 10,
            "batch_size": 32
        }
    }
    training_result = await mas.agents["ml_engineer"].train_model(training_spec)
    print(f"Training result: {training_result}")

if __name__ == "__main__":
    asyncio.run(run_demo()) 