from src.pipeline.orchestrator import Orchestrator

if __name__ == "__main__":
    orchestrator = Orchestrator(explainable=True)
    orchestrator.run(
        image_a_path="./images/dev-473-3-img0.png",
        image_b_path="./images/dev-473-3-img1.png",
        question="What is uniquely similar about these two images?",
        answer_options=["group 1", "group 2"]
    )
    