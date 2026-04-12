# BarrierBench: Evaluating Large Language Models for Safety Verification in Dynamical Systems
<a href='https://hycodev.com/dataset/barrierbench'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2511.09363'><img src='https://img.shields.io/badge/L4DC-2026-blue'></a>

[Ali Taheri](https://alitaheri2002.github.io/), [Alireza Taban](https://www.linkedin.com/in/alireza-taban-90a460121/), [Sadegh Soudjani](https://hycodev.com/ssoudjani), [Ashutosh Trivedi](https://www.cs.colorado.edu/~trivedi/)

Isfahan University of Technology, Max Planck Institute for Software Systems, University of Colorado Boulder

-----
### Brief Introduction
This paper, accepted at the **8th Annual Learning for Dynamics & Control (L4DC) 2026**, introduces BarrierBench, a benchmark of 100 dynamical systems for evaluating LLMs on safety verification via barrier certificate synthesis. We propose an LLM-based agentic framework that uses natural language reasoning to propose, refine, and validate barrier certificates, integrating SMT-based verification and retrieval-augmented generation. The framework achieves over 90% success in generating valid certificates.

## Benchmark
The benchmark is available at: [https://hycodev.com/dataset/barrierbench](https://hycodev.com/data/BarrierBench.json)

## Installation
```bash
pip install anthropic sympy z3-solver numpy
```

## Overview
Our agentic framework addresses the limitations of classical barrier certificate synthesis by:
1. **Retrieval-Augmented Generation**: Retrieve similar solved problems from the benchmark dataset
2. **Barrier Synthesis Agent**: LLM-guided template discovery and candidate generation
3. **Barrier Verifier Agent**: SMT-based formal verification of candidate certificates
4. **Iterative Refinement**: Feedback loop to refine candidates based on verification results

## Usage
### Setup
Set your API key in `main.py`:
```python
synthesizer = BarrierSynthesisAgent(api_key="YOUR_API_KEY", max_iterations=5, dataset_json_path="../Benchmark/barrier_dataset.json")
```

### Running
```python
python main.py
```

### Input Format
```json
{
  "problem": {
    "dynamics": "mathematical equations",
    "initial_set": {"type", "radius"/"bounds", "center"},
    "unsafe_set": {"type", "radius"/"bounds", "complement"},
    "controller_parameters": "control inputs (if applicable)"
  },
  "barrier": "barrier function polynomial",
  "controllers": "control law expressions",
  "template_type": "solution classification"
}
```

## Results

| Approach | Claude Sonnet 4 | ChatGPT-4o |
|----------|----------------|------------|
| Baseline (Single Prompt) | 41% | 17% |
| **Full Framework** | **90%** | **46%** |
| Improvement | +49% | +29% |

## Citation
```bibtex
@inproceedings{taheri2026barrierbench,
  title={BarrierBench: Evaluating Large Language Models for Safety Verification in Dynamical Systems},
  author={Taheri, Ali and Taban, Alireza and Soudjani, Sadegh and Trivedi, Ashutosh},
  booktitle={8th Annual Learning for Dynamics \& Control (L4DC)},
  year={2026}
}
```
