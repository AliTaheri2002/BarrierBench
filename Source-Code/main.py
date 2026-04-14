import logging
from barrier_synthesis_agent import BarrierSynthesisAgent

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO)



if __name__ == "__main__":

    test_problem = {
    "dynamics": "dx1/dt = x2 + u0, dx2/dt = -0.8*x1 - 0.15*x2 + 0.2*x1**2 + u1, dx3/dt = -0.1*x3 + 0.1*x1 + u2",
    "initial_set": {
      "type": "ball",
      "radius": 0.4,
      "center": [0, 0, 0]
    },
    "unsafe_set": {
      "type": "ball",
      "radius": 3,
      "center": [0, 0, 0],
      "complement": True
    },
    "controller_parameters": "u0, u1, u2"
}

    synthesizer = BarrierSynthesisAgent(api_key="YOUR API KEY", max_iterations=5, dataset_json_path="../Benchmark/BarrierBench.json")

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