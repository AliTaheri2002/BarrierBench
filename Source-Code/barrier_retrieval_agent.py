import os
import json
import re
import logging
import anthropic
import sympy as sp
from typing import Optional

logger = logging.getLogger(__name__)


class BarrierRetrievalAgent:
    def __init__(self, json_file_path="barrier_dataset.json"):
        self.json_file_path = json_file_path
        self.test_cases = []
        self.client: Optional[anthropic.Anthropic] = None
        self.model: str = "claude-sonnet-4-20250514"
        self._load()

    def _load(self):
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    self.test_cases = json.loads(content).get('solved_problems', [])
                else:
                    self.test_cases = []
                    self._save()
        else:
            self.test_cases = []
            self._save()

    def _save(self):
        with open(self.json_file_path, 'w', encoding='utf-8') as f:
            json.dump({'solved_problems': self.test_cases}, f, indent=2)

    def store(self, problem, barrier_certificate, template_type, controller_certificate=None):
        record = {'problem': problem, 'barrier': barrier_certificate, 'template_type': template_type}
        if controller_certificate is not None:
            record['controllers'] = controller_certificate
        self.test_cases.append(record)
        self._save()

    def find_most_similar(self, target_problem):
        target_features = self._extract_features(target_problem)
        if not target_features:
            return None
        compatible = [c for c in self.test_cases if self._check_compatible(target_features, self._extract_features(c['problem']))]

        if not compatible:
            return None
        return self._llm_select(target_problem, compatible)

    def _extract_features(self, problem):
        dynamics = problem.get('dynamics', '')
        initial_set = problem.get('initial_set', {})
        unsafe_set = problem.get('unsafe_set', {})

        return { 'dimension'   : self._get_dimension(dynamics), 'time_domain' : self._get_time_domain(dynamics),
                 'linearity'   : self._get_linearity(dynamics), 'set_topology': self._get_topology(initial_set, unsafe_set)}

    def _get_dimension(self, dynamics):
        return len(set(re.findall(r'x(\d+)', dynamics)))

    def _get_time_domain(self, dynamics):
        if '[k+1]' in dynamics or '[k]' in dynamics:
            return 'discrete'
        elif 'dt' in dynamics or 'd/dt' in dynamics:
            return 'continuous'
        return 'unknown'

    def _get_linearity(dynamics: str) -> str:
        var_names = sorted(set(re.findall(r'x\d+', dynamics)))
        if not var_names:
            return 'unknown'
        
        rhs_match = re.search(r'=\s*(.+)', dynamics)
        rhs = rhs_match.group(1) if rhs_match else dynamics
        
        expr = sp.sympify(rhs, {n: sp.Symbol(n) for n in var_names})
        
        if any((isinstance(node, sp.Pow) and node.args[1] != 1) or 
            (isinstance(node, sp.Function) and node.args) or
            (isinstance(node, sp.Mul) and len([v for v in var_names if node.has(sp.Symbol(v))]) > 1)
            for node in sp.preorder_traversal(expr)):
            return 'nonlinear'
        
        return 'linear'

    def _get_topology(self, initial_set, unsafe_set):
        init_type = initial_set.get('type', 'unknown')
        unsafe_type = unsafe_set.get('type', 'unknown')
        unsafe_complement = unsafe_set.get('complement', False)

        if unsafe_type == 'union':
            inner_types = [s.get('type', 'unknown') for s in unsafe_set.get('sets', [])]
            unsafe_type = f"union_of_{'_'.join(set(inner_types))}"

        return f"{init_type}_to_{'complement_' if unsafe_complement else ''}{unsafe_type}"

    def _check_compatible(self, f1, f2):
        if not f2:
            return False
        checks = [f1.get('dimension') == f2.get('dimension'), f1.get('time_domain') == f2.get('time_domain'),
                  f1.get('linearity') == f2.get('linearity'), self._topology_compatible(f1.get('set_topology'), f2.get('set_topology'))]
        return all(checks)

    def _topology_compatible(self, t1, t2):
        if t1 == t2:
            return True
        return t1.replace('complement_', '') == t2.replace('complement_', '')

    def _llm_select(self, target_problem, compatible):
        candidates_text = ""
        for i, c in enumerate(compatible, 1):
            p = c['problem']
            candidates_text += f"\nCANDIDATE {i}:\n"
            candidates_text += f"Dynamics: {p.get('dynamics')}\n"
            candidates_text += f"Initial: {p.get('initial_set')}\n"
            candidates_text += f"Unsafe: {p.get('unsafe_set')}\n"
            candidates_text += f"Successful barrier: {c['barrier']}\n"

        prompt = f"""TARGET PROBLEM:
Dynamics: {target_problem.get('dynamics')}
Initial set: {target_problem.get('initial_set')}
Unsafe set: {target_problem.get('unsafe_set')}

COMPATIBLE CANDIDATES (all are fundamentally similar):{candidates_text}

Which candidate has the most similar problem type and structure to the target problem?
Focus on: system structure, problem type, and mathematical pattern similarity.

Answer with only the candidate number (1, 2, 3, etc.): """

        response = self.client.messages.create(model=self.model, max_tokens=100, messages=[{"role": "user", "content": prompt}])

        raw_output = response.content[0].text.strip()

        match = re.search(r'\b(\d+)\b', raw_output)
        if match:
            index = int(match.group(1))
            if 1 <= index <= len(compatible):
                return compatible[index - 1]
            else:
                return compatible[0]
        logger.warning("LLM failed to select a candidate")
        return None
