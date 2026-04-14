import os
import re
import z3
import time
import logging
import tempfile
import anthropic
import subprocess
import sympy as sp
from barrier_parsing import parse_barrier_certificate


logger = logging.getLogger(__name__)




SOLVER_SELECTION_PROMPT = """Select the best SMT solver based on barrier expiration and the dynamic system.

PROBLEM:
Dynamics: {dynamics}
Barrier: {barrier}

AVAILABLE SOLVERS:
- cvc5
- z3
- yices

Without long explanations, give a short and precise answer.

Format your response as:
SOLVER: [solver name]"""


TIMEOUT_ANALYSIS_PROMPT = """Solver {solver_name} timed out after {timeout_ms}ms.

PROBLEM:
Dynamics: {dynamics}
Barrier: {barrier}

Given the above information, should we attempt again with more time, or is this barrier too complex to verify?

Without long explanations, give a short and precise answer.

Format your response as:
RETRY: yes or no
TIMEOUT_MULTIPLIER: [number] (only if RETRY is yes, e.g., 1.5 or 2.0)"""


ERROR_ANALYSIS_PROMPT = """Solver {solver_name} failed during verification.

ERROR: {error_type}
MESSAGE: {error_msg}

PROBLEM:
Dynamics: {dynamics}
Barrier: {barrier}

REMAINING SOLVERS:
{remaining_solvers}

Select a different solver to try.

Without long explanations, give a short and precise answer.

Format your response as:
NEXT_SOLVER: [solver name]"""




class AgenticSMTVerifier:
    def __init__(self, anthropic_client: anthropic.Anthropic, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic_client
        self.model = model
        self.available_solvers = ['z3', 'cvc5', 'yices']
        self.attempt_history = []

    def verify(self, barrier_expr, initial_set, unsafe_set, dynamics):
        self.attempt_history = []
        problem_summary = {'dynamics': dynamics, 'barrier': barrier_expr}

        selected_solver = self._llm_select_solver(problem_summary)
        if not selected_solver:
            logger.warning("LLM failed to select solver, using default z3")
            selected_solver = 'z3'


        result = self._execute_solver(selected_solver, barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms=30000)

        if result['success']:
            return result

        if result.get('error_type') == 'timeout':
            retry_decision = self._llm_analyze_timeout(selected_solver, 30000, problem_summary)
            if retry_decision.get('retry'):
                new_timeout = int(30000 * retry_decision.get('multiplier', 1.5))
                result = self._execute_solver(selected_solver, barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms=new_timeout)
                if result['success']:
                    return result

        if result.get('error_type') in ['unknown', 'timeout', 'parse_error']:
            remaining = [s for s in self.available_solvers if s != selected_solver]
            if remaining:
                next_solver = self._llm_suggest_next_solver(selected_solver, result, problem_summary, remaining)
                if next_solver and next_solver in remaining:
                    result2 = self._execute_solver(next_solver, barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms=30000)
                    if result2['success']:
                        return result2

        return {'success': False, 'error': 'All solver attempts failed', 'attempts': self.attempt_history}

    def _llm_select_solver(self, problem):
        prompt = SOLVER_SELECTION_PROMPT.format(dynamics=problem['dynamics'], barrier=problem['barrier'])

        response = self.client.messages.create(model=self.model, max_tokens=250, messages=[{"role": "user", "content": prompt}])
        content = response.content[0].text.strip()

        match = re.search(r'SOLVER:\s*(\w+)', content, re.IGNORECASE)
        if match:
            solver = match.group(1).lower()
            if solver in self.available_solvers:
                return solver
        return None

    def _llm_analyze_timeout(self, solver_name, timeout_ms, problem):
        prompt = TIMEOUT_ANALYSIS_PROMPT.format(solver_name=solver_name, timeout_ms=timeout_ms,
                                                dynamics=problem['dynamics'], barrier=problem['barrier'])

        response = self.client.messages.create(model=self.model, max_tokens=250,messages=[{"role": "user", "content": prompt}])
        content = response.content[0].text.strip()

        retry_match = re.search(r'RETRY:\s*(yes|no)', content, re.IGNORECASE)
        multiplier_match = re.search(r'TIMEOUT_MULTIPLIER:\s*([\d.]+)', content, re.IGNORECASE)

        retry = retry_match.group(1).lower() == 'yes' if retry_match else False
        multiplier = float(multiplier_match.group(1)) if multiplier_match else 1.5

        return {'retry': retry, 'multiplier': multiplier}

    def _llm_suggest_next_solver(self, failed_solver, result, problem, remaining):
        prompt = ERROR_ANALYSIS_PROMPT.format(solver_name=failed_solver,error_type=result.get('error_type', 'unknown'),
                 error_msg=result.get('error', 'No details'), dynamics=problem['dynamics'],
                 barrier=problem['barrier'], remaining_solvers=', '.join(remaining))

        response = self.client.messages.create(model=self.model, max_tokens=250, messages=[{"role": "user", "content": prompt}])
        content = response.content[0].text.strip()

        match = re.search(r'NEXT_SOLVER:\s*(\w+)', content, re.IGNORECASE)
        if match:
            solver = match.group(1).lower()
            if solver in remaining:
                return solver
        return None

    def _execute_solver(self, solver_name, barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms):
        start_time = time.time()

        if solver_name == 'z3':
            result = self._verify_with_z3(barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms)
        elif solver_name == 'cvc5':
            result = self._verify_with_cvc5(barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms)
        elif solver_name == 'yices':
            result = self._verify_with_yices(barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms)
        else:
            result = {'success': False, 'error': f'Unknown solver: {solver_name}', 'error_type': 'unknown'}

        self.attempt_history.append({'solver': solver_name, 'result': result, 'time': time.time() - start_time, 'timeout': timeout_ms})

        return result

    def _verify_with_z3(self, barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms):
        barrier_expression, var = parse_barrier_certificate(barrier_expr)

        if barrier_expression is None:
            return {'success': False, 'error': 'Failed to parse barrier', 'error_type': 'parse_error'}

        try:
            conditions = smt_based_verification(barrier_expression, var, initial_set, unsafe_set, dynamics, timeout_ms)
            return {'success': True, 'barrier_expression': str(barrier_expression),
                'var': [str(v) for v in var], 'conditions': conditions,
                'all_satisfied': conditions.get('all_conditions_satisfied', False)
            }
        except TimeoutError:
            return {'success': False, 'error': 'Z3 verification timeout', 'error_type': 'timeout'}
        except Exception as e:
            error_str = str(e).lower()
            error_type = 'timeout' if 'timeout' in error_str else 'unknown'
            return {'success': False, 'error': str(e), 'error_type': error_type}

    def _verify_with_cvc5(self, barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms):
        barrier_expression, var = parse_barrier_certificate(barrier_expr)

        if barrier_expression is None:
            return {'success': False, 'error': 'Failed to parse barrier', 'error_type': 'parse_error'}

        smtlib_content = self._generate_smtlib2(barrier_expression, var, initial_set, unsafe_set, dynamics, timeout_ms, solver_name='cvc5')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
            f.write(smtlib_content)
            temp_file = f.name

        try:
            result = subprocess.run(['cvc5', '--incremental', '--lang=smt2', temp_file], capture_output=True, text=True, timeout=timeout_ms / 1000.0)
            output = result.stdout.strip()

            conditions = self._parse_smtlib_results(output)

            return {'success': True, 'barrier_expression': str(barrier_expression),
                'var': [str(v) for v in var], 'conditions': conditions,
                'all_satisfied': conditions.get('all_conditions_satisfied', False)
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'cvc5 verification timeout', 'error_type': 'timeout'}
        except FileNotFoundError:
            logger.warning("cvc5 not found, falling back to Z3")
            return self._verify_with_z3(barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms)
        except Exception as e:
            error_type = 'timeout' if 'timeout' in str(e).lower() else 'unknown'
            return {'success': False, 'error': str(e), 'error_type': error_type}
        finally:
            os.unlink(temp_file)

    def _verify_with_yices(self, barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms):
        barrier_expression, var = parse_barrier_certificate(barrier_expr)

        if barrier_expression is None:
            return {'success': False, 'error': 'Failed to parse barrier', 'error_type': 'parse_error'}

        smtlib_content = self._generate_smtlib2(barrier_expression, var, initial_set, unsafe_set, dynamics, timeout_ms, solver_name='yices')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
            f.write(smtlib_content)
            temp_file = f.name

        try:
            result = subprocess.run(['yices-smt2', '--incremental', temp_file], capture_output=True, text=True, timeout=timeout_ms / 1000.0)
            output = result.stdout.strip()

            conditions = self._parse_smtlib_results(output)
            return {'success': True, 'barrier_expression': str(barrier_expression),
                    'var': [str(v) for v in var], 'conditions': conditions,'all_satisfied': conditions.get('all_conditions_satisfied', False)}
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Yices verification timeout', 'error_type': 'timeout'}
        except FileNotFoundError:
            logger.warning("yices-smt2 not found, falling back to Z3")
            return self._verify_with_z3(barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms)
        except Exception as e:
            error_type = 'timeout' if 'timeout' in str(e).lower() else 'unknown'
            return {'success': False, 'error': str(e), 'error_type': error_type}
        finally:
            os.unlink(temp_file)

    def _generate_smtlib2(self, barrier_expression, variables, initial_set, unsafe_set, dynamics, timeout_ms, solver_name='cvc5'):
        lines = []

        has_transcendental = any(
            isinstance(expr, (sp.sin, sp.cos, sp.exp))
            for expr in sp.preorder_traversal(barrier_expression)
        )

        dynamics_exprs = parse_dynamics(dynamics, list(barrier_expression.free_symbols))
        if dynamics_exprs:
            for dyn_expr in dynamics_exprs:
                if any(isinstance(e, (sp.sin, sp.cos, sp.exp)) for e in sp.preorder_traversal(dyn_expr)):
                    has_transcendental = True
                    break

        if solver_name == 'yices':
            lines.append("(set-logic QF_NRA)")
            use_approximation = True
        elif solver_name == 'cvc5':
            lines.append("(set-logic ALL)" if has_transcendental else "(set-logic QF_NRA)")
            use_approximation = False
        else:
            lines.append("(set-logic QF_NRA)")
            use_approximation = has_transcendental

        lines.append(f"(set-option :timeout {timeout_ms})")

        for var in variables:
            lines.append(f"(declare-fun {str(var)} () Real)")

        to_smtlib = self._sympy_to_smtlib_with_taylor if use_approximation else self._sympy_to_smtlib
        barrier_smtlib = to_smtlib(barrier_expression)

        # Condition 1
        lines.append("; Condition 1: B(x) <= 0 for all x in initial set")
        lines.append("(push 1)")
        for c in self._set_to_smtlib_constraints(initial_set, variables):
            lines.append(f"(assert {c})")
        lines.append(f"(assert (> {barrier_smtlib} 0))")
        lines.append("(check-sat)")
        lines.append("(pop 1)")

        # Condition 2
        lines.append("; Condition 2: B(x) > 0 for all x in unsafe set")
        lines.append("(push 1)")
        for c in self._set_to_smtlib_constraints(unsafe_set, variables):
            lines.append(f"(assert {c})")
        lines.append(f"(assert (<= {barrier_smtlib} 0))")
        lines.append("(check-sat)")
        lines.append("(pop 1)")

        # Condition 3
        lines.append("; Condition 3: Invariance condition")
        lines.append("(push 1)")

        all_vars = sorted(list(barrier_expression.free_symbols), key=lambda x: int(str(x)[1:]) if str(x)[1:].isdigit() else 999)
        is_discrete = isinstance(dynamics, str) and ('[k+1]' in dynamics or '[k]' in dynamics)

        if is_discrete:
            subs_dict = {v: dynamics_exprs[i] for i, v in enumerate(all_vars)}
            barrier_next = barrier_expression.subs(subs_dict)
            barrier_next_smtlib = to_smtlib(barrier_next)
            lines.append(f"(assert (> (- {barrier_next_smtlib} {barrier_smtlib}) 0))")
        else:
            gradient = [sp.diff(barrier_expression, v) for v in all_vars]
            lie = sum(g * d for g, d in zip(gradient, dynamics_exprs))
            lie_smtlib = to_smtlib(lie)
            lines.append(f"(assert (= {barrier_smtlib} 0))")
            lines.append(f"(assert (>= {lie_smtlib} 0))")

        lines.append("(check-sat)")
        lines.append("(pop 1)")
        lines.append("(exit)")

        smtlib = '\n'.join(lines)

        return smtlib

    def _sympy_to_smtlib(self, expr):
        def convert(e):
            if isinstance(e, sp.Symbol):
                return str(e)
            elif isinstance(e, sp.Number):
                val = float(e)
                return f"(- {abs(val)})" if val < 0 else str(val)
            elif isinstance(e, sp.Add):
                args = [convert(a) for a in e.args]
                return args[0] if len(args) == 1 else f"(+ {' '.join(args)})"
            elif isinstance(e, sp.Mul):
                args = [convert(a) for a in e.args]
                return args[0] if len(args) == 1 else f"(* {' '.join(args)})"
            elif isinstance(e, sp.Pow):
                base = convert(e.args[0])
                try:
                    exp_int = int(e.args[1])
                    if exp_int == 2:
                        return f"(* {base} {base})"
                    elif exp_int == 3:
                        return f"(* {base} (* {base} {base}))"
                    elif exp_int == 4:
                        return f"(* {base} (* {base} (* {base} {base})))"
                    else:
                        return f"(^ {base} {exp_int})"
                except:
                    return f"(^ {base} {convert(e.args[1])})"
            elif isinstance(e, sp.sin):
                return f"(sin {convert(e.args[0])})"
            elif isinstance(e, sp.cos):
                return f"(cos {convert(e.args[0])})"
            else:
                return str(e)
        return convert(expr)

    def _sympy_to_smtlib_with_taylor(self, expr):
        def convert(e):
            if isinstance(e, sp.Symbol):
                return str(e)
            elif isinstance(e, sp.Number):
                val = float(e)
                return f"(- {abs(val)})" if val < 0 else str(val)
            elif isinstance(e, sp.Add):
                args = [convert(a) for a in e.args]
                return args[0] if len(args) == 1 else f"(+ {' '.join(args)})"
            elif isinstance(e, sp.Mul):
                args = [convert(a) for a in e.args]
                return args[0] if len(args) == 1 else f"(* {' '.join(args)})"
            elif isinstance(e, sp.Pow):
                base = convert(e.args[0])
                try:
                    exp_int = int(e.args[1])
                    if exp_int == 2:
                        return f"(* {base} {base})"
                    elif exp_int == 3:
                        return f"(* {base} (* {base} {base}))"
                    elif exp_int == 4:
                        return f"(* {base} (* {base} (* {base} {base})))"
                    else:
                        return f"(^ {base} {exp_int})"
                except:
                    return f"(^ {base} {convert(e.args[1])})"
            elif isinstance(e, sp.sin):
                x = convert(e.args[0])
                x2 = f"(* {x} {x})"
                x3 = f"(* {x2} {x})"
                x5 = f"(* {x3} {x2})"
                return f"(- (+ {x} (/ {x5} 120.0)) (/ {x3} 6.0))"
            elif isinstance(e, sp.cos):
                x = convert(e.args[0])
                x2 = f"(* {x} {x})"
                x4 = f"(* {x2} {x2})"
                return f"(- (+ 1.0 (/ {x4} 24.0)) (/ {x2} 2.0))"
            else:
                return str(e)
        return convert(expr)

    def _set_to_smtlib_constraints(self, set_description, variables):
        constraints = []
        set_type = set_description.get('type')

        if set_type == 'ball':
            radius = set_description.get('radius')
            center = set_description.get('center')
            is_complement = set_description.get('complement', False)
            terms = []
            for i, var in enumerate(variables):
                cv = center[i] if i < len(center) else 0
                if cv == 0:
                    terms.append(f"(* {str(var)} {str(var)})")
                else:
                    terms.append(f"(* (- {str(var)} {cv}) (- {str(var)} {cv}))")
            sq = f"(+ {' '.join(terms)})" if len(terms) > 1 else terms[0]
            op = ">=" if is_complement else "<="
            constraints.append(f"({op} {sq} {radius**2})")

        elif set_type in ('bounds', 'box'):
            bounds = set_description.get('bounds', [])
            is_complement = set_description.get('complement', False)
            if is_complement:
                outside = [f"(or (< {str(v)} {low}) (> {str(v)} {high}))" for v, (low, high) in zip(variables, bounds)]
                if outside:
                    constraints.append(f"(or {' '.join(outside)})")
            else:
                for var, (low, high) in zip(variables, bounds):
                    constraints.append(f"(>= {str(var)} {low})")
                    constraints.append(f"(<= {str(var)} {high})")

        elif set_type == 'union':
            sub_constraints = []
            for subset in set_description.get('sets', []):
                sc = self._set_to_smtlib_constraints(subset, variables)
                if sc:
                    sub_constraints.append(f"(and {' '.join(sc)})" if len(sc) > 1 else sc[0])
            if sub_constraints:
                constraints.append(f"(or {' '.join(sub_constraints)})")

        return constraints

    def _parse_smtlib_results(self, output):
        lines = output.strip().split('\n')
        sat_results = [l.strip() for l in lines if l.strip() in ('sat', 'unsat', 'unknown')]

        results = {'condition_1': False, 'condition_2': False, 'condition_3': False, 'verification_details': {}}

        if len(sat_results) >= 3:
            for i, key in enumerate(['condition_1', 'condition_2', 'condition_3']):
                results[key] = (sat_results[i] == 'unsat')
                results['verification_details'][key] = {'satisfied': results[key], 'result': sat_results[i]}

        results['all_conditions_satisfied'] = all([results['condition_1'], results['condition_2'], results['condition_3']])
        return results




def smt_based_verification(barrier_expression, variables, initial_set, unsafe_set, dynamics, timeout_ms=100000):

    z3_variables = {str(v): z3.Real(str(v)) for v in variables}
    barrier_z3 = convert_sympy_to_z3(barrier_expression, z3_variables)

    results = {'condition_1': False, 'condition_2': False, 'condition_3': False, 'z3_verification': True, 'verification_details': {}}

    c1 = verify_initial_condition(barrier_z3, z3_variables, initial_set, timeout_ms)
    results['condition_1'] = c1['satisfied']
    results['verification_details']['condition_1'] = c1

    c2 = verify_unsafe_condition(barrier_z3, z3_variables, unsafe_set, timeout_ms)
    results['condition_2'] = c2['satisfied']
    results['verification_details']['condition_2'] = c2

    c3 = verify_invariance_condition(barrier_expression, barrier_z3, z3_variables, dynamics, timeout_ms)
    results['condition_3'] = c3['satisfied']
    results['verification_details']['condition_3'] = c3

    all_sat = all([results['condition_1'], results['condition_2'], results['condition_3']])
    results['all_conditions_satisfied'] = all_sat

    if all_sat:
        print("  SMT: All 3 conditions satisfied")
    else:
        failed = [f"Condition {i+1}" for i, ok in enumerate([results['condition_1'], results['condition_2'], results['condition_3']]) if not ok]
        print(f"  SMT: Failed conditions: {', '.join(failed)}")

    return results


def convert_sympy_to_z3(sympy_expr, z3_variables):

    expr_str = str(sympy_expr)

    namespace = {**z3_variables, '__builtins__': {}}
    for i in range(1, 11):
        var_name = f'x{i}'
        if var_name not in namespace:
            namespace[var_name] = z3_variables.get(var_name, z3.Real(var_name))

    expr_str = re.sub(r'(\w+)\*\*(\d+)', r'_pow_(\1, \2)', expr_str)
    expr_str = re.sub(r'sin\(([^)]+)\)', r'_sin_(\1)', expr_str)
    expr_str = re.sub(r'cos\(([^)]+)\)', r'_cos_(\1)', expr_str)
    expr_str = re.sub(r'exp\(([^)]+)\)', r'_exp_(\1)', expr_str)

    def _pow_(base, exp):
        exp = int(exp)
        result = base
        for _ in range(exp - 1):
            result = result * base
        return result

    def _sin_(x):
        x2 = x * x; x3 = x2 * x; x5 = x3 * x2; x7 = x5 * x2
        return x - x3/6 + x5/120 - x7/5040

    def _cos_(x):
        x2 = x * x; x4 = x2 * x2; x6 = x4 * x2
        return 1 - x2/2 + x4/24 - x6/720

    def _exp_(x):
        x2 = x * x; x3 = x2 * x; x4 = x2 * x2
        return 1 + x + x2/2 + x3/6 + x4/24

    namespace.update({'_pow_': _pow_, '_sin_': _sin_, '_cos_': _cos_, '_exp_': _exp_})

    z3_expr = eval(expr_str, namespace)
    return z3_expr


def verify_initial_condition(barrier_z3, z3_variables, initial_set, timeout_ms=30000):
    solver = z3.Solver()
    initial_constraints = get_set_constraints(initial_set, z3_variables)
    for c in initial_constraints:
        solver.add(c)
    solver.add(barrier_z3 > 0)
    solver.set("timeout", timeout_ms)

    result = solver.check()
    if result == z3.unsat:
        return {'satisfied': True, 'method': 'z3_formal', 'details': 'No counterexample found'}
    elif result == z3.sat:
        model = solver.model()
        counterexample = {str(v): model[z3v] for v, z3v in z3_variables.items() if model[z3v] is not None}
        return {'satisfied': False, 'method': 'z3_formal', 'details': 'Counterexample found', 'counterexample': counterexample}
    else:
        raise TimeoutError("Z3 returned unknown for initial condition")


def verify_unsafe_condition(barrier_z3, z3_variables, unsafe_set, timeout_ms=30000):
    solver = z3.Solver()
    unsafe_constraints = get_set_constraints(unsafe_set, z3_variables)
    for c in unsafe_constraints:
        solver.add(c)
    solver.add(barrier_z3 <= 0)
    solver.set("timeout", timeout_ms)

    result = solver.check()
    if result == z3.unsat:
        return {'satisfied': True, 'method': 'z3_formal', 'details': 'No counterexample found'}
    elif result == z3.sat:
        model = solver.model()
        counterexample = {str(v): model[z3v] for v, z3v in z3_variables.items() if model[z3v] is not None}
        return {'satisfied': False, 'method': 'z3_formal', 'details': 'Counterexample found', 'counterexample': counterexample}
    else:
        raise TimeoutError("Z3 returned unknown for unsafe condition")


def verify_invariance_condition(barrier_expression, barrier_z3, z3_variables, dynamics, timeout_ms=30000):
    if not isinstance(dynamics, str):
        raise TypeError("Invalid dynamics type - expected str")

    is_discrete = '[k+1]' in dynamics or '[k]' in dynamics

    barrier_symbols = sorted(barrier_expression.free_symbols, key=lambda s: int(str(s)[1:]) if str(s).startswith('x') and str(s)[1:].isdigit() else 999)

    dynamics_temp = re.sub(r'\[k\+1\]', '', dynamics)
    dynamics_temp = re.sub(r'\[k\]', '', dynamics_temp)
    system_dimension = len([eq for eq in dynamics_temp.split(',') if eq.strip()])

    all_system_variables = []
    for idx in range(1, system_dimension + 1):
        var_name = f'x{idx}'
        existing = next((s for s in barrier_symbols if str(s) == var_name), None)
        all_system_variables.append(existing if existing else sp.Symbol(var_name, real=True))

    dynamics_exprs = parse_dynamics(dynamics, all_system_variables)
    if dynamics_exprs is None or len(dynamics_exprs) != len(all_system_variables):
        raise ValueError(f'Dynamics dimension mismatch')

    controller_symbols = set()
    for dy in dynamics_exprs:
        controller_symbols.update(s for s in dy.free_symbols if str(s).startswith('u') and (str(s)[1:].isdigit() or len(str(s)) == 1))
    for s in controller_symbols:
        if str(s) not in z3_variables:
            z3_variables[str(s)] = z3.Real(str(s))

    for v in all_system_variables:
        if str(v) not in z3_variables:
            z3_variables[str(v)] = z3.Real(str(v))

    if is_discrete:
        subs_dict = {v: dynamics_exprs[i] for i, v in enumerate(all_system_variables)}
        barrier_next = barrier_expression.subs(subs_dict)
        barrier_next_z3 = convert_sympy_to_z3(barrier_next, z3_variables)

        solver = z3.Solver()
        solver.add(barrier_next_z3 - barrier_z3 > 0)
        solver.set("timeout", timeout_ms)
        result = solver.check()

        if result == z3.unsat:
            return {'satisfied': True, 'method': 'z3_formal', 'details': 'No counterexample found'}
        elif result == z3.sat:
            model = solver.model()
            counterexample = {str(v): model[z3v] for v, z3v in z3_variables.items() if model[z3v] is not None}
            return {'satisfied': False, 'method': 'z3_formal', 'details': 'Counterexample found', 'counterexample': counterexample}
        else:
            raise TimeoutError("Z3 returned unknown for discrete invariance condition")
    else:
        gradient = [sp.diff(barrier_expression, v) for v in all_system_variables]

        if all(g == 0 for g in gradient):
            raise ValueError("Gradient is all zeros")

        lie_derivative = sum(g * d for g, d in zip(gradient, dynamics_exprs))
        lie_z3 = convert_sympy_to_z3(lie_derivative, z3_variables)

        solver = z3.Solver()
        solver.add(barrier_z3 == 0)
        solver.add(lie_z3 >= 0)
        solver.set("timeout", timeout_ms)
        result = solver.check()

        if result == z3.unsat:
            return {'satisfied': True, 'method': 'z3_formal', 'details': 'No counterexample found'}
        elif result == z3.sat:
            model = solver.model()
            counterexample = {str(v): model[z3v] for v, z3v in z3_variables.items() if model[z3v] is not None}
            return {'satisfied': False, 'method': 'z3_formal', 'details': 'Counterexample found', 'counterexample': counterexample}
        else:
            raise TimeoutError("Z3 returned unknown for continuous invariance condition")


def get_set_constraints(set_description, z3_variables):
    constraints = []
    set_type = set_description.get('type')

    if set_type == 'union':
        sets = set_description.get('sets', [])
        if not sets:
            logger.error("Empty sets in union")
            return None
        union_constraints = []
        for subset in sets:
            sc = get_set_constraints(subset, z3_variables)
            if sc:
                union_constraints.append(z3.And(*sc))
        if union_constraints:
            constraints.append(z3.Or(*union_constraints))
        return constraints

    var_list = sorted(z3_variables.items(), key=lambda x: int(x[0][1:]) if x[0].startswith('x') and x[0][1:].isdigit() else 999)

    if set_type == 'ball':
        radius = set_description.get('radius')
        center = set_description.get('center')
        is_complement = set_description.get('complement', False)

        if len(center) != len(z3_variables):
            logger.error("center and z3 variables mismatch")
            return None

        sq_dist = sum((z3v - (center[i] if i < len(center) else 0)) ** 2 for i, (_, z3v) in enumerate(var_list))
        constraints.append(sq_dist >= radius**2 if is_complement else sq_dist <= radius**2)

    elif set_type in ('bounds', 'box'):
        bounds = set_description.get('bounds')
        is_complement = set_description.get('complement', False)

        if is_complement:
            outside = [z3.Or(z3v < bounds[i][0], z3v > bounds[i][1]) for i, (_, z3v) in enumerate(var_list) if i < len(bounds)]
            if outside:
                constraints.append(z3.Or(*outside))
        else:
            for i, (_, z3v) in enumerate(var_list):
                if i < len(bounds):
                    constraints.append(z3v >= bounds[i][0])
                    constraints.append(z3v <= bounds[i][1])
    else:
        logger.error(f"Unknown or missing set type: {set_type}")
        return None

    return constraints


def parse_dynamics(system_dynamics, variables):
    if not isinstance(system_dynamics, str):
        logger.error("Invalid dynamics type - expected str")
        return None

    system_dynamics = re.sub(r'\[k\+1\]', '', system_dynamics)
    system_dynamics = re.sub(r'\[k\]', '', system_dynamics)

    exprs = []
    for eq in [e.strip() for e in system_dynamics.split(',')]:
        rhs = eq.split('=')[-1].strip() if '=' in eq else eq
        for var in variables:
            rhs = re.sub(r'\b' + str(var) + r'\b', str(var), rhs)
        exprs.append(sp.sympify(rhs))

    if len(exprs) != len(variables):
        logger.error(f"Got {len(exprs)} equations but need {len(variables)}")
        return None

    return exprs

def validate_barrier_with_agentic_smt(synthesized_barrier, initial_set, unsafe_set, dynamics, anthropic_client, model="claude-sonnet-4-20250514"):
    verifier = AgenticSMTVerifier(anthropic_client, model)
    result = verifier.verify(synthesized_barrier, initial_set, unsafe_set, dynamics)
    return result