from src.pipeline.orchestrator import Orchestrator

if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run(
        image_a_path="./images/dev-364-0-img0.png",
        image_b_path="./images/dev-364-0-img1.png",
        question="Which group of women are stronger?",
        answer_options=["group 1", "group 2"]
    )
    