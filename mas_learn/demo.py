import asyncio
from mas_learn.orchestrator import MultiAgentOrchestrator


async def run_enhanced_demo():
    # Initialize orchestrator
    mas = MultiAgentOrchestrator()

    print("\n=== Enhanced Multi-Agent System Demo ===")

    # 1. Parallel Research and Analysis
    research_task = {
        "objective": "novel deep learning model with Pytorch for predicting survival of passengers on the titanic",
        "requirements": {
            "technical_depth": "detailed",
            "implementation_focus": True,
            "ethical_considerations": False,
        },
    }

    # Run parallel tasks
    tasks = [
        mas.agents["researcher"].web_search(
            f"latest research {research_task['objective']}"
        ),
        mas.agents["ml_engineer"].analyze_architecture_options(research_task),
        mas.agents["coder"].prepare_implementation_plan(research_task),
    ]

    print("\nInitiating parallel research and analysis...")
    results = await asyncio.gather(*tasks)

    # 2. Collaborative Solution Development
    print("\nStarting collaborative solution development...")

    # Researcher analyzes findings and proposes approach
    research_findings = await mas.agents["researcher"].synthesize_findings(results[0])
    print("\nResearch Synthesis Complete - Awaiting your review:")
    user_approval = await mas.agents["researcher"].report_to_user(
        {"phase": "research_synthesis", "findings": research_findings}
    )

    if user_approval.lower() != "yes":
        print("Research direction not approved. Adjusting approach...")
        return

    # ML Engineer designs architecture based on research
    architecture = await mas.agents["ml_engineer"].design_architecture(
        {"research_findings": research_findings, "technical_requirements": results[1]}
    )

    # Coder implements solution with ML Engineer's guidance
    max_recovery_attempts = 3
    attempt_count = 0

    while attempt_count < max_recovery_attempts:
        try:
            implementation = await mas.agents["coder"].implement_solution(
                {"architecture": architecture, "implementation_plan": results[2]}
            )

            if implementation["status"] == "recovered":
                print("\nRecovered from implementation errors:")
                print("Changes made:", implementation["recovery_changes"])

            # Execute and validate implementation
            print("\nValidating implementation...")
            validation_result = await mas.agents["executor"].execute_code(
                implementation.get("code", "")
            )

            # If we get here without errors, break the loop
            break

        except Exception as e:
            attempt_count += 1
            if attempt_count >= max_recovery_attempts:
                print(
                    f"\nFinal implementation failed after {max_recovery_attempts} attempts: {str(e)}"
                )
                print("Please review the logs for details and try again.")
                return
            print(f"\nAttempt {attempt_count} failed, trying recovery...")

    # Final results printing with error handling
    print("\nFinal Results:")
    print("-" * 80)
    print(f"Research Findings: {research_findings.get('summary', research_findings)}")
    print(f"Architecture Design: {architecture.get('design_summary', architecture)}")
    print(f"Implementation Status: {validation_result}")
    print("-" * 80)


if __name__ == "__main__":
    asyncio.run(run_enhanced_demo())
