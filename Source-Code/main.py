import logging
from barrier_synthesis_agent import BarrierSynthesisAgent

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    test_problem = {
        "dynamics": "dx1/dt = -0.1*x1 - 0.5",
        "initial_set": {
            "type": "bounds",
            "bounds": [[1.0, 2.0]]
        },
        "unsafe_set": {
            "type": "bounds",
            "bounds": [[5.0, 6.0]],
            "complement": False
        }
    }

    synthesizer = BarrierSynthesisAgent(api_key="API KEY", max_iterations=5, dataset_json_path="../Benchmark/barrier_dataset.json")

    result = synthesizer.synthesize_barrier_certificate(test_problem)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    if result['success']:
        print(f" SUCCESS: {result['barrier_certificate']}")
        print(f"Found in iteration: {result['iteration_found']}")
    else:
        print(f" FAILED - Best score: {result.get('best_score', 0)}/3")
        if result.get('best_barrier'):
            print(f"Best attempt: {result['best_barrier']}")

    print(f"Time: {result['total_time']:.2f}s")
