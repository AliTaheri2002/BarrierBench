import sympy as sp
from typing import Dict, List, Union, Any, Optional
import re
import logging
import z3
import time
import json
import anthropic
from barrier_parsing import parse_barrier_certificate, clean_barrier_expression
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)


# ============================================================================
# PROMPTS
# ============================================================================

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


# ============================================================================
# AGENTIC SMT VERIFIER
# ============================================================================

class AgenticSMTVerifier:
    def __init__(self, anthropic_client: anthropic.Anthropic, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic_client
        self.model = model
        self.available_solvers = ['z3', 'cvc5', 'yices']
        self.attempt_history = []
        
    def verify(self, barrier_expr: str, initial_set: Dict, unsafe_set: Dict, dynamics: Union[str, Dict, List]) -> Dict[str, Any]:
        """Main agentic verification entry point"""
        
        self.attempt_history = []
        problem_summary = {
            'dynamics': dynamics,
            'barrier': barrier_expr
        }
        
        # Step 1: LLM selects initial solver
        selected_solver = self._llm_select_solver(problem_summary)
        if not selected_solver:
            logger.warning("LLM failed to select solver, using default z3")
            selected_solver = 'z3'
        
        logger.info(f"LLM selected solver: {selected_solver}")
        
        # Step 2: Try first solver
        result = self._execute_solver(selected_solver, barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms=30000)
        
        if result['success']:
            return result
        
        # Step 3: Analyze failure
        if result.get('error_type') == 'timeout':
            retry_decision = self._llm_analyze_timeout(selected_solver, 30000, problem_summary)
            
            if retry_decision.get('retry'):
                new_timeout = int(30000 * retry_decision.get('multiplier', 1.5))
                logger.info(f"Retrying {selected_solver} with timeout {new_timeout}ms")
                result = self._execute_solver(selected_solver, barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms=new_timeout)
                
                if result['success']:
                    return result
        
        # Step 4: Try second solver if first failed
        if result.get('error_type') in ['unknown', 'timeout', 'parse_error']:
            used_solvers = [selected_solver]
            remaining = [s for s in self.available_solvers if s not in used_solvers]
            
            if remaining:
                next_solver = self._llm_suggest_next_solver(selected_solver, result, problem_summary, remaining)
                
                if next_solver and next_solver in remaining:
                    logger.info(f"Trying second solver: {next_solver}")
                    result2 = self._execute_solver(next_solver, barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms=30000)
                    
                    if result2['success']:
                        return result2
        
        # All attempts failed
        return {
            'success': False,
            'error': 'All solver attempts failed',
            'attempts': self.attempt_history
        }
    
    def _llm_select_solver(self, problem: Dict) -> Optional[str]:
        """Ask LLM to select best solver"""
        try:
            prompt = SOLVER_SELECTION_PROMPT.format(
                dynamics=problem['dynamics'],
                barrier=problem['barrier']
            )
            
            # ========== ADD THIS ==========
            print("\n" + "="*60)
            print("LLM SOLVER SELECTION PROMPT:")
            print("="*60)
            print(prompt)
            print("="*60)
            # ==============================
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=250,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text.strip()
            
            # ========== ADD THIS ==========
            print("\n" + "="*60)
            print("LLM SOLVER SELECTION RESPONSE:")
            print("="*60)
            print(content)
            print("="*60)
            # ==============================
            
            match = re.search(r'SOLVER:\s*(\w+)', content, re.IGNORECASE)
            
            if match:
                solver = match.group(1).lower()
                if solver in self.available_solvers:
                    return solver
            
            return None
            
        except Exception as e:
            logger.error(f"LLM solver selection failed: {e}")
            return None
    
    def _llm_analyze_timeout(self, solver_name: str, timeout_ms: int, problem: Dict) -> Dict:
        """Ask LLM if we should retry with more time"""
        try:
            prompt = TIMEOUT_ANALYSIS_PROMPT.format(
                solver_name=solver_name,
                timeout_ms=timeout_ms,
                dynamics=problem['dynamics'],
                barrier=problem['barrier']
            )
            
            # ========== ADD THIS ==========
            print("\n" + "="*60)
            print("LLM TIMEOUT ANALYSIS PROMPT:")
            print("="*60)
            print(prompt)
            print("="*60)
            # ==============================
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=250,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text.strip()
            
            # ========== ADD THIS ==========
            print("\n" + "="*60)
            print("LLM TIMEOUT ANALYSIS RESPONSE:")
            print("="*60)
            print(content)
            print("="*60)
            # ==============================
            
            retry_match = re.search(r'RETRY:\s*(yes|no)', content, re.IGNORECASE)
            multiplier_match = re.search(r'TIMEOUT_MULTIPLIER:\s*([\d.]+)', content, re.IGNORECASE)
            
            retry = retry_match.group(1).lower() == 'yes' if retry_match else False
            multiplier = float(multiplier_match.group(1)) if multiplier_match else 1.5
            
            return {'retry': retry, 'multiplier': multiplier}
            
        except Exception as e:
            logger.error(f"LLM timeout analysis failed: {e}")
            return {'retry': False}
    
    def _llm_suggest_next_solver(self, failed_solver: str, result: Dict, problem: Dict, remaining: List[str]) -> Optional[str]:
        """Ask LLM which solver to try next"""
        try:
            prompt = ERROR_ANALYSIS_PROMPT.format(
                solver_name=failed_solver,
                error_type=result.get('error_type', 'unknown'),
                error_msg=result.get('error', 'No details'),
                dynamics=problem['dynamics'],
                barrier=problem['barrier'],
                remaining_solvers=', '.join(remaining)
            )
            
            # ========== ADD THIS ==========
            print("\n" + "="*60)
            print("LLM ERROR ANALYSIS PROMPT:")
            print("="*60)
            print(prompt)
            print("="*60)
            # ==============================
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=250,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text.strip()
            
            # ========== ADD THIS ==========
            print("\n" + "="*60)
            print("LLM ERROR ANALYSIS RESPONSE:")
            print("="*60)
            print(content)
            print("="*60)
            # ==============================
            
            match = re.search(r'NEXT_SOLVER:\s*(\w+)', content, re.IGNORECASE)
            
            if match:
                solver = match.group(1).lower()
                if solver in remaining:
                    return solver
            
            return None
            
        except Exception as e:
            logger.error(f"LLM next solver suggestion failed: {e}")
            return None
    
    def _execute_solver(self, solver_name: str, barrier_expr: str, initial_set: Dict, 
                       unsafe_set: Dict, dynamics: Union[str, Dict, List], timeout_ms: int) -> Dict[str, Any]:
        """Execute verification with specific solver"""
        
        start_time = time.time()
        
        try:
            if solver_name == 'z3':
                result = self._verify_with_z3(barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms)
            elif solver_name == 'cvc5':
                result = self._verify_with_cvc5(barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms)
            elif solver_name == 'yices':
                result = self._verify_with_yices(barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms)
            else:
                result = {
                    'success': False,
                    'error': f'Unknown solver: {solver_name}',
                    'error_type': 'unknown'
                }
            
            elapsed = time.time() - start_time
            
            self.attempt_history.append({
                'solver': solver_name,
                'result': result,
                'time': elapsed,
                'timeout': timeout_ms
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Solver {solver_name} execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'unknown'
            }
    
    def _verify_with_z3(self, barrier_expr: str, initial_set: Dict, unsafe_set: Dict, 
                       dynamics: Union[str, Dict, List], timeout_ms: int) -> Dict[str, Any]:
        """Verify using Z3 solver"""
        try:
            barrier_expression, var = parse_barrier_certificate(barrier_expr)
            
            if barrier_expression is None:
                return {
                    'success': False,
                    'error': 'Failed to parse barrier',
                    'error_type': 'parse_error'
                }
            
            conditions = smt_based_verification(barrier_expression, var, initial_set, unsafe_set, dynamics, timeout_ms)
            
            return {
                'success': True,
                'barrier_expression': str(barrier_expression),
                'var': [str(v) for v in var],
                'conditions': conditions,
                'all_satisfied': conditions.get('all_conditions_satisfied', False)
            }
            
        except TimeoutError:
            return {
                'success': False,
                'error': 'Z3 verification timeout',
                'error_type': 'timeout'
            }
        except Exception as e:
            error_str = str(e).lower()
            if 'timeout' in error_str:
                return {'success': False, 'error': str(e), 'error_type': 'timeout'}
            else:
                return {'success': False, 'error': str(e), 'error_type': 'unknown'}
    
    def _verify_with_cvc5(self, barrier_expr: str, initial_set: Dict, unsafe_set: Dict, 
                 dynamics: Union[str, Dict, List], timeout_ms: int) -> Dict[str, Any]:
        try:
            barrier_expression, var = parse_barrier_certificate(barrier_expr)
            
            if barrier_expression is None:
                return {'success': False, 'error': 'Failed to parse barrier', 'error_type': 'parse_error'}
            
            # Pass solver_name='cvc5'
            smtlib_content = self._generate_smtlib2(barrier_expression, var, initial_set, unsafe_set, dynamics, timeout_ms, solver_name='cvc5')
                
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
                f.write(smtlib_content)
                temp_file = f.name
            
            try:
                # Run cvc5
                result = subprocess.run(
                    ['cvc5', '--incremental', '--lang=smt2', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout_ms / 1000.0
                )
                
                output = result.stdout.strip()
                
                print("\n" + "="*60)
                print("cvc5 RAW OUTPUT:")
                print("="*60)
                print(output)
                print("="*60)
                print("cvc5 STDERR:")
                print("="*60)
                print(result.stderr)
                print("="*60)
                
                # Parse results
                conditions = self._parse_smtlib_results(output)
                
                return {
                    'success': True,
                    'barrier_expression': str(barrier_expression),
                    'var': [str(v) for v in var],
                    'conditions': conditions,
                    'all_satisfied': conditions.get('all_conditions_satisfied', False)
                }
                
            finally:
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'cvc5 verification timeout',
                'error_type': 'timeout'
            }
        except FileNotFoundError:
            logger.warning("cvc5 not found, falling back to Z3")
            return self._verify_with_z3(barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms)
        except Exception as e:
            error_str = str(e).lower()
            if 'timeout' in error_str:
                return {'success': False, 'error': str(e), 'error_type': 'timeout'}
            else:
                return {'success': False, 'error': str(e), 'error_type': 'unknown'}
    
    def _verify_with_yices(self, barrier_expr: str, initial_set: Dict, unsafe_set: Dict, 
                  dynamics: Union[str, Dict, List], timeout_ms: int) -> Dict[str, Any]:
        try:
            barrier_expression, var = parse_barrier_certificate(barrier_expr)
            
            if barrier_expression is None:
                return {'success': False, 'error': 'Failed to parse barrier', 'error_type': 'parse_error'}
            
            # Pass solver_name='yices'
            smtlib_content = self._generate_smtlib2(barrier_expression, var, initial_set, unsafe_set, dynamics, timeout_ms, solver_name='yices')
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
                f.write(smtlib_content)
                temp_file = f.name
            
            try:
                # Run yices-smt2
                result = subprocess.run(
                     ['yices-smt2', '--incremental', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout_ms / 1000.0
                )
                
                output = result.stdout.strip()
                
                # ===== DEBUG =====
                print("\n" + "="*60)
                print("yices RAW OUTPUT:")
                print("="*60)
                print(output)
                print("="*60)
                print("yices STDERR:")
                print("="*60)
                print(result.stderr)
                print("="*60)
                # =================

                # Parse results
                conditions = self._parse_smtlib_results(output)
                
                return {
                    'success': True,
                    'barrier_expression': str(barrier_expression),
                    'var': [str(v) for v in var],
                    'conditions': conditions,
                    'all_satisfied': conditions.get('all_conditions_satisfied', False)
                }
                
            finally:
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Yices verification timeout',
                'error_type': 'timeout'
            }
        except FileNotFoundError:
            logger.warning("yices-smt2 not found, falling back to Z3")
            return self._verify_with_z3(barrier_expr, initial_set, unsafe_set, dynamics, timeout_ms)
        except Exception as e:
            error_str = str(e).lower()
            if 'timeout' in error_str:
                return {'success': False, 'error': str(e), 'error_type': 'timeout'}
            else:
                return {'success': False, 'error': str(e), 'error_type': 'unknown'}

    def _generate_smtlib2(self, barrier_expression: sp.Expr, variables: List[sp.Symbol],
                 initial_set: Dict, unsafe_set: Dict, dynamics: Union[str, Dict, List],
                 timeout_ms: int, solver_name: str = 'cvc5') -> str:
        """Generate SMT-LIB2 format for barrier verification"""
        
        lines = []
        
        # Check if we have transcendental functions
        has_transcendental = any(
            isinstance(expr, (sp.sin, sp.cos, sp.exp)) 
            for expr in sp.preorder_traversal(barrier_expression)
        )
        
        dynamics_exprs = parse_dynamics(dynamics, list(barrier_expression.free_symbols))
        if dynamics_exprs:
            for dyn_expr in dynamics_exprs:
                if any(isinstance(expr, (sp.sin, sp.cos, sp.exp)) for expr in sp.preorder_traversal(dyn_expr)):
                    has_transcendental = True
                    break
        
        # Choose logic based on solver
        if solver_name == 'yices':
            # Yices only supports QF_NRA, so we must use Taylor approximation
            lines.append("(set-logic QF_NRA)")
            use_approximation = True
        elif solver_name == 'cvc5':
            # cvc5 supports ALL logic with transcendentals
            if has_transcendental:
                lines.append("(set-logic ALL)")
            else:
                lines.append("(set-logic QF_NRA)")
            use_approximation = False
        else:
            lines.append("(set-logic QF_NRA)")
            use_approximation = has_transcendental
        
        lines.append(f"(set-option :timeout {timeout_ms})")
        
        # Declare variables
        for var in variables:
            lines.append(f"(declare-fun {str(var)} () Real)")
        
        # Convert barrier to SMT-LIB2
        if use_approximation:
            barrier_smtlib = self._sympy_to_smtlib_with_taylor(barrier_expression)
        else:
            barrier_smtlib = self._sympy_to_smtlib(barrier_expression)
        
        # Condition 1
        lines.append("; Condition 1: B(x) <= 0 for all x in initial set")
        lines.append("(push 1)")
        initial_constraints = self._set_to_smtlib_constraints(initial_set, variables)
        for constraint in initial_constraints:
            lines.append(f"(assert {constraint})")
        lines.append(f"(assert (> {barrier_smtlib} 0))")
        lines.append("(check-sat)")
        lines.append("(pop 1)")
        
        # Condition 2
        lines.append("; Condition 2: B(x) > 0 for all x in unsafe set")
        lines.append("(push 1)")
        unsafe_constraints = self._set_to_smtlib_constraints(unsafe_set, variables)
        for constraint in unsafe_constraints:
            lines.append(f"(assert {constraint})")
        lines.append(f"(assert (<= {barrier_smtlib} 0))")
        lines.append("(check-sat)")
        lines.append("(pop 1)")
        
        # Condition 3
        lines.append("; Condition 3: Invariance condition")
        lines.append("(push 1)")
        
        all_system_variables = sorted(list(barrier_expression.free_symbols), key=lambda x: int(str(x)[1:]) if str(x)[1:].isdigit() else 999)
        is_discrete = isinstance(dynamics, str) and ('[k+1]' in dynamics or '[k]' in dynamics)
        
        if is_discrete:
            subs_dict = {var: dynamics_exprs[i] for i, var in enumerate(all_system_variables)}
            barrier_next = barrier_expression.subs(subs_dict)
            if use_approximation:
                barrier_next_smtlib = self._sympy_to_smtlib_with_taylor(barrier_next)
            else:
                barrier_next_smtlib = self._sympy_to_smtlib(barrier_next)
            lines.append(f"(assert (> (- {barrier_next_smtlib} {barrier_smtlib}) 0))")
        else:
            gradient = [sp.diff(barrier_expression, var) for var in all_system_variables]
            lie_derivative = sum(grad * dyn for grad, dyn in zip(gradient, dynamics_exprs))
            if use_approximation:
                lie_derivative_smtlib = self._sympy_to_smtlib_with_taylor(lie_derivative)
            else:
                lie_derivative_smtlib = self._sympy_to_smtlib(lie_derivative)
            lines.append(f"(assert (= {barrier_smtlib} 0))")
            lines.append(f"(assert (>= {lie_derivative_smtlib} 0))")
        
        lines.append("(check-sat)")
        lines.append("(pop 1)")
        lines.append("(exit)")

        result = '\n'.join(lines)
        
        print("\n" + "="*60)
        print(f"GENERATED SMT-LIB2 for {solver_name}:")
        print("="*60)
        print(result)
        print("="*60)
        
        return result

    def _sympy_to_smtlib(self, expr: sp.Expr) -> str:
        """Convert SymPy expression to SMT-LIB2 format"""
        
        def convert_expr(e):
            if isinstance(e, sp.Symbol):
                return str(e)
            elif isinstance(e, sp.Number):
                val = float(e)
                if val < 0:
                    return f"(- {abs(val)})"  # <-- FIX: -6.0 → (- 6.0)
                else:
                    return str(val)
            elif isinstance(e, sp.Add):
                args = [convert_expr(arg) for arg in e.args]
                if len(args) == 1:
                    return args[0]
                return f"(+ {' '.join(args)})"
            elif isinstance(e, sp.Mul):
                args = [convert_expr(arg) for arg in e.args]
                if len(args) == 1:
                    return args[0]
                return f"(* {' '.join(args)})"
            elif isinstance(e, sp.Pow):
                base = convert_expr(e.args[0])
                exp = convert_expr(e.args[1])
                try:
                    exp_int = int(e.args[1])
                    if exp_int == 2:
                        return f"(* {base} {base})"
                    elif exp_int == 3:
                        return f"(* {base} (* {base} {base}))"
                    elif exp_int == 4:
                        return f"(* {base} (* {base} (* {base} {base})))"
                    else:
                        return f"(^ {base} {exp})"
                except:
                    return f"(^ {base} {exp})"
            elif isinstance(e, sp.sin):
                arg = convert_expr(e.args[0])
                return f"(sin {arg})"
            elif isinstance(e, sp.cos):
                arg = convert_expr(e.args[0])
                return f"(cos {arg})"
            else:
                return str(e)
        
        return convert_expr(expr)

    def _sympy_to_smtlib_with_taylor(self, expr: sp.Expr) -> str:
        """Convert SymPy with Taylor approximation for sin/cos"""
        
        def convert_expr(e):
            if isinstance(e, sp.Symbol):
                return str(e)
            elif isinstance(e, sp.Number):
                val = float(e)
                if val < 0:
                    return f"(- {abs(val)})"
                else:
                    return str(val)
            elif isinstance(e, sp.Add):
                args = [convert_expr(arg) for arg in e.args]
                if len(args) == 1:
                    return args[0]
                return f"(+ {' '.join(args)})"
            elif isinstance(e, sp.Mul):
                args = [convert_expr(arg) for arg in e.args]
                if len(args) == 1:
                    return args[0]
                return f"(* {' '.join(args)})"
            elif isinstance(e, sp.Pow):
                base = convert_expr(e.args[0])
                exp = convert_expr(e.args[1])
                try:
                    exp_int = int(e.args[1])
                    if exp_int == 2:
                        return f"(* {base} {base})"
                    elif exp_int == 3:
                        return f"(* {base} (* {base} {base}))"
                    elif exp_int == 4:
                        return f"(* {base} (* {base} (* {base} {base})))"
                    else:
                        return f"(^ {base} {exp})"
                except:
                    return f"(^ {base} {exp})"
            elif isinstance(e, sp.sin):
                # Taylor: sin(x) ≈ x - x³/6 + x⁵/120
                arg = e.args[0]
                x = convert_expr(arg)
                x2 = f"(* {x} {x})"
                x3 = f"(* {x2} {x})"
                x5 = f"(* {x3} {x2})"
                return f"(- (+ {x} (/ {x5} 120.0)) (/ {x3} 6.0))"
            elif isinstance(e, sp.cos):
                # Taylor: cos(x) ≈ 1 - x²/2 + x⁴/24
                arg = e.args[0]
                x = convert_expr(arg)
                x2 = f"(* {x} {x})"
                x4 = f"(* {x2} {x2})"
                return f"(- (+ 1.0 (/ {x4} 24.0)) (/ {x2} 2.0))"
            else:
                return str(e)
        
        return convert_expr(expr)

    def _set_to_smtlib_constraints(self, set_description: Dict, variables: List[sp.Symbol]) -> List[str]:
        """Convert set description to SMT-LIB2 constraints"""
        constraints = []
        
        if set_description.get('type') == 'ball':
            radius = set_description.get('radius')
            center = set_description.get('center')
            is_complement = set_description.get('complement', False)
            
            # Build squared distance
            terms = []
            for i, var in enumerate(variables):
                center_val = center[i] if i < len(center) else 0
                if center_val == 0:
                    terms.append(f"(* {str(var)} {str(var)})")
                else:
                    terms.append(f"(* (- {str(var)} {center_val}) (- {str(var)} {center_val}))")
            
            squared_dist = ' '.join(terms)
            if len(terms) > 1:
                squared_dist = f"(+ {squared_dist})"
            else:
                squared_dist = terms[0]
            
            radius_squared = radius ** 2
            
            if is_complement:
                constraints.append(f"(>= {squared_dist} {radius_squared})")
            else:
                constraints.append(f"(<= {squared_dist} {radius_squared})")
        
        return constraints


    def _parse_smtlib_results(self, output: str) -> Dict[str, Any]:
        """Parse SMT-LIB2 solver output"""
        lines = output.strip().split('\n')
        
        results = {
            'condition_1': False,
            'condition_2': False,
            'condition_3': False,
            'verification_details': {}
        }
        
        # Filter only sat/unsat/unknown results, ignore other lines
        sat_results = [line.strip() for line in lines 
                    if line.strip() in ['sat', 'unsat', 'unknown']]
        
        if len(sat_results) >= 3:
            # Condition 1: unsat means no counterexample (good)
            results['condition_1'] = (sat_results[0] == 'unsat')
            results['verification_details']['condition_1'] = {
                'satisfied': results['condition_1'],
                'result': sat_results[0]
            }
            
            # Condition 2: unsat means no counterexample (good)
            results['condition_2'] = (sat_results[1] == 'unsat')
            results['verification_details']['condition_2'] = {
                'satisfied': results['condition_2'],
                'result': sat_results[1]
            }
            
            # Condition 3: unsat means no counterexample (good)
            results['condition_3'] = (sat_results[2] == 'unsat')
            results['verification_details']['condition_3'] = {
                'satisfied': results['condition_3'],
                'result': sat_results[2]
            }
        
        results['all_conditions_satisfied'] = all([
            results['condition_1'],
            results['condition_2'],
            results['condition_3']
        ])
        
        return results
# ============================================================================
# ORIGINAL SMT VERIFICATION (Modified to accept timeout)
# ============================================================================

def smt_based_verification(barrier_expression: sp.Expr, variables: List[sp.Symbol], 
                                initial_set: Dict, unsafe_set: Dict, 
                                dynamics: Union[str, Dict, List], timeout_ms: int = 100000) -> Dict[str, Any]:
    
    logger.info(f"Starting formal SMT verification with timeout {timeout_ms}ms")

    try:
        
        z3_variables = {}
        for var in variables:
            z3_variables[str(var)] = z3.Real(str(var))
        
        barrier_z3 = convert_sympy_to_z3(barrier_expression, z3_variables)
        
        results = {
            'condition_1': False,
            'condition_2': False, 
            'condition_3': False,
            'z3_verification': True,
            'verification_details': {}
        }
        
        logger.info("Checking Condition 1: B(x) ≤ 0 for all x ∈ X₀")
        initial_condition_result = verify_initial_condition(barrier_z3, z3_variables, initial_set, timeout_ms)
        results['condition_1'] = initial_condition_result['satisfied']
        results['verification_details']['condition_1'] = initial_condition_result
        
        logger.info("Checking Condition 2: B(x) > 0 for all x ∈ Xᵘ")
        unsafe_condition_result = verify_unsafe_condition(barrier_z3, z3_variables, unsafe_set, timeout_ms)
        results['condition_2'] = unsafe_condition_result['satisfied']
        results['verification_details']['condition_2'] = unsafe_condition_result
        
        logger.info("Checking Condition 3: ∇B(x)·f(x) < 0 or B(f(x)) - B(x) ≤ 0")
        invariance_condition_result = verify_invariance_condition(barrier_expression, barrier_z3, z3_variables, dynamics, timeout_ms)
        results['condition_3'] = invariance_condition_result['satisfied']
        results['verification_details']['condition_3'] = invariance_condition_result
        
        all_satisfied = all([results['condition_1'], results['condition_2'], results['condition_3']])
        results['all_conditions_satisfied'] = all_satisfied
        
        if all_satisfied:
            logger.info("SUCCESS: ALL 3 CONDITIONS SATISFIED!")
        else:
            failed_conditions = []
            for i, satisfied in enumerate([results['condition_1'], results['condition_2'], results['condition_3']], 1):
                if not satisfied:
                    failed_conditions.append(f"Condition {i}")
            logger.info(f"Failed conditions: {', '.join(failed_conditions)}")
        
        satisfied_count = sum([results['condition_1'], results['condition_2'], results['condition_3']])
        logger.info(f"SMT Verification completed: {satisfied_count}/3 conditions satisfied")
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"SMT verification failed: {e}")


def convert_sympy_to_z3(sympy_expr: sp.Expr, z3_variables: Dict):
    
    logger.info(f"Converting expression to Z3 format.")

    try:
        expr_str = str(sympy_expr)
        
        namespace = {
            **z3_variables,
            '__builtins__': {},
        }
        
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
            if exp == 1:
                return base
            elif exp == 2:
                return base * base
            elif exp == 3:
                return base * base * base
            elif exp == 4:
                return base * base * base * base
            elif exp == 5:
                return base * base * base * base * base
            else:
                result = base
                for _ in range(exp - 1):
                    result = result * base
                return result
        
        def _sin_(x):
            x2 = x * x
            x3 = x2 * x
            x5 = x3 * x2
            x7 = x5 * x2
            return x - x3/6 + x5/120 - x7/5040
        
        def _cos_(x):
            x2 = x * x
            x4 = x2 * x2
            x6 = x4 * x2 
            return 1 - x2/2 + x4/24 - x6/720
        
        def _exp_(x):
            x2 = x * x
            x3 = x2 * x
            x4 = x2 * x2
            return 1 + x + x2/2 + x3/6 + x4/24
        
        namespace['_pow_'] = _pow_
        namespace['_sin_'] = _sin_
        namespace['_cos_'] = _cos_
        namespace['_exp_'] = _exp_
        
        z3_expr = eval(expr_str, namespace)
        logger.info(f"Successfully converted to Z3: {z3_expr}")
        return z3_expr
        
    except Exception as e:
        raise RuntimeError(f"Cannot convert expression to Z3: {e}")


def verify_initial_condition(barrier_z3, z3_variables: Dict, initial_set: Dict, timeout_ms: int = 30000) -> Dict[str, Any]:
    
    MAX = 3
    
    for i in range(MAX):
        try:
            solver = z3.Solver()
            
            initial_constraints = get_set_constraints(initial_set, z3_variables)
            for constraint in initial_constraints:
                solver.add(constraint)
            
            solver.add(barrier_z3 > 0)
            
            solver.set("timeout", timeout_ms)
            result = solver.check()
            
            if result == z3.unsat:
                return {
                    'satisfied': True,
                    'method': 'z3_formal',
                    'details': 'No counterexample found'
                }
            elif result == z3.sat:
                model = solver.model()
                counterexample = {str(var): model[z3_var] for var, z3_var in z3_variables.items() if model[z3_var] is not None}
                return {
                    'satisfied': False,
                    'method': 'z3_formal',
                    'details': 'Counterexample found',
                    'counterexample': counterexample
                }
            else:
                if i < MAX - 1:
                    continue
                else:
                    raise TimeoutError("Z3 solver returned unknown after retries")
                    
        except Exception as e:
            if i < MAX - 1:
                continue
            else:
                raise RuntimeError(f"Initial condition verification failed: {e}")


def verify_unsafe_condition(barrier_z3, z3_variables: Dict, unsafe_set: Dict, timeout_ms: int = 30000) -> Dict[str, Any]:
    
    MAX = 3
    
    for i in range(MAX):
        try:
            solver = z3.Solver()
            
            unsafe_constraints = get_set_constraints(unsafe_set, z3_variables)
            for constraint in unsafe_constraints:
                solver.add(constraint)
            
            solver.add(barrier_z3 <= 0)
            
            solver.set("timeout", timeout_ms)
            result = solver.check()
            
            if result == z3.unsat:
                return {
                    'satisfied': True,
                    'method': 'z3_formal',
                    'details': 'No counterexample found'
                }
            elif result == z3.sat:
                model = solver.model()
                counterexample = {str(var): model[z3_var] for var, z3_var in z3_variables.items() if model[z3_var] is not None}
                return {
                    'satisfied': False,
                    'method': 'z3_formal',
                    'details': 'Counterexample found',
                    'counterexample': counterexample
                }
            else:
                if i < MAX - 1:
                    continue
                else:
                    raise TimeoutError("Z3 solver returned unknown after retries")
                    
        except Exception as e:
            if i < MAX - 1:
                continue
            else:
                raise RuntimeError(f"Unsafe condition verification failed: {e}")


def verify_invariance_condition(barrier_expression: sp.Expr, barrier_z3, z3_variables: Dict, 
                                dynamics: Union[str, Dict, List], timeout_ms: int = 30000) -> Dict[str, Any]:
    
    MAX = 3
    
    for i in range(MAX):
        try:
            dynamics_exprs = parse_dynamics(dynamics, list(barrier_expression.free_symbols))
            
            if dynamics_exprs is None:
                raise ValueError("Failed to parse dynamics")
            
            all_system_variables = sorted(list(barrier_expression.free_symbols), key=lambda x: int(str(x)[1:]) if str(x)[1:].isdigit() else 999)
            
            is_discrete = isinstance(dynamics, str) and ('[k+1]' in dynamics or '[k]' in dynamics)
            
            controller_symbols = set()
            for dyn_expr in dynamics_exprs:
                controller_symbols.update(dyn_expr.free_symbols - set(all_system_variables))
            
            for ctrl_sym in controller_symbols:
                ctrl_name = str(ctrl_sym)
                if ctrl_name not in z3_variables:
                    z3_variables[ctrl_name] = z3.Real(ctrl_name)
                    logger.info(f"Added controller parameter to Z3: {ctrl_name}")
            
            if is_discrete:
                logger.info("Using discrete-time barrier condition: B(f(x)) - B(x) ≤ 0")
                
                subs_dict = {var: dynamics_exprs[i] for i, var in enumerate(all_system_variables)}
                barrier_next = barrier_expression.subs(subs_dict)
                
                barrier_next_z3 = convert_sympy_to_z3(barrier_next, z3_variables)                
                solver = z3.Solver()
                solver.add(barrier_next_z3 - barrier_z3 > 0)
                
                solver.set("timeout", timeout_ms)
                result = solver.check()
                
                if result == z3.unsat:
                    return {
                        'satisfied': True,
                        'method': 'z3_formal',
                        'details': 'No counterexample found'
                    }
                elif result == z3.sat:
                    model = solver.model()
                    counterexample = {str(var): model[z3_var] for var, z3_var in z3_variables.items() if model[z3_var] is not None}
                    return {
                        'satisfied': False,
                        'method': 'z3_formal',
                        'details': 'Counterexample found',
                        'counterexample': counterexample
                    }
                else:
                    if i < MAX - 1:
                        continue
                    else:
                        raise TimeoutError("Z3 solver returned unknown after retries")
            
            else:
                logger.info("Using continuous-time barrier condition: ∇B(x)·f(x) < 0")
                
                gradient = [sp.diff(barrier_expression, var) for var in all_system_variables]
                
                if all(g == 0 for g in gradient):
                    raise ValueError("Gradient is all zeros")
                
                lie_derivative = sum(grad * dyn for grad, dyn in zip(gradient, dynamics_exprs))
                
                lie_derivative_z3 = convert_sympy_to_z3(lie_derivative, z3_variables)
                
                solver = z3.Solver()                
                solver.add(barrier_z3 == 0)
                solver.add(lie_derivative_z3 >= 0)                
                solver.set("timeout", timeout_ms)

                result = solver.check()
                
                if result == z3.unsat:
                    return {
                        'satisfied': True,
                        'method': 'z3_formal',
                        'details': 'No counterexample found'
                    }
                elif result == z3.sat:
                    model = solver.model()
                    counterexample = {str(var): model[z3_var] for var, z3_var in z3_variables.items() if model[z3_var] is not None}
                    return {
                        'satisfied': False,
                        'method': 'z3_formal',
                        'details': 'Counterexample found',
                        'counterexample': counterexample
                    }
                else:
                    if i < MAX - 1:
                        continue
                    else:
                        raise TimeoutError("Z3 solver returned unknown after retries")
                    
        except Exception as e:
            if i < MAX - 1:
                continue
            else:
                raise RuntimeError(f"Invariance condition verification failed: {e}")


def get_set_constraints(set_description: Dict, z3_variables: Dict) -> List:
    constraints = []
    try:
        if set_description.get('type') == 'ball':
            radius = set_description.get('radius')
            center = set_description.get('center')
            is_complement = set_description.get('complement', False)
            
            if len(center) != len(z3_variables):
                logger.error(f"center and z3 variables mismatch")
                return None
            
            squared_distance = 0
            var_list = sorted(z3_variables.items(), key=lambda x: int(x[0][1:]) if x[0].startswith('x') and x[0][1:].isdigit() else 999)
            
            for i, (var_name, z3_var) in enumerate(var_list):
                center_coord = center[i] if i < len(center) else 0
                squared_distance += (z3_var - center_coord) ** 2
            
            if is_complement:
                constraints.append(squared_distance >= radius ** 2)
            else:
                constraints.append(squared_distance <= radius ** 2)
        
        elif 'bounds' in set_description:
            bounds = set_description['bounds']
            var_list = sorted(z3_variables.items(), key=lambda x: int(x[0][1:]) if x[0].startswith('x') and x[0][1:].isdigit() else 999)
            
            for i, (var_name, z3_var) in enumerate(var_list):
                if i < len(bounds):
                    low, high = bounds[i]
                    constraints.append(z3_var >= low)
                    constraints.append(z3_var <= high)
        else:
            logger.error(f"Unknown or missing set type")
            return None
    
    except Exception as e:
        logger.error(f"Error creating Z3 constraints: {e}")
        return None 
    
    return constraints


def parse_dynamics(system_dynamics: str, variables: List[sp.Symbol]) -> List[sp.Expr]:
    
    if not isinstance(system_dynamics, str):
        logger.error(f"Invalid dynamics type")
        return None

    try:
        system_dynamics = re.sub(r'\[k\+1\]', '', system_dynamics)
        system_dynamics = re.sub(r'\[k\]', '', system_dynamics)
        
        system_equations = [eq.strip() for eq in system_dynamics.split(',')]
        
        exprs = []
        for eq in system_equations:
            if '=' in eq:
                rhs = eq.split('=')[-1].strip()
            else:
                rhs = eq.strip()
            
            for var in variables:
                var_name = str(var)
                rhs = re.sub(r'\b' + var_name + r'\b', str(var), rhs)
            
            expr = sp.sympify(rhs)
            exprs.append(expr)
        
        if len(exprs) != len(variables):
            print(f"ERROR: Got {len(exprs)} equations but need {len(variables)}")
            return None

        return exprs
        
    except Exception as e:
        logger.error(f"Error parsing dynamics: {e}")
        return None


# ============================================================================
# PUBLIC API
# ============================================================================

def validate_barrier_with_smt(synthesized_barrier: str, initial_set: Dict, unsafe_set: Dict, 
                                 dynamics: Union[str, Dict, List]) -> Dict[str, Any]:
    """Original SMT validation (non-agentic)"""
    try:
        barrier_expression, var = parse_barrier_certificate(synthesized_barrier)

        if barrier_expression is None:
            return {'success': False, 'error': 'Failed to parse barrier'}
        
        conditions = smt_based_verification(barrier_expression, var, initial_set, unsafe_set, dynamics)
        
        return {
            'success': True,
            'barrier_expression': str(barrier_expression),
            'var': [str(v) for v in var],
            'conditions': conditions,
            'all_satisfied': conditions.get('all_conditions_satisfied', False)
        }
        
    except Exception as e:
        print("Verification failed:", str(e))
        return {'success': False, 'error': str(e)}


def validate_barrier_with_agentic_smt(synthesized_barrier: str, initial_set: Dict, unsafe_set: Dict, 
                                       dynamics: Union[str, Dict, List],
                                       anthropic_client: anthropic.Anthropic,
                                       model: str = "claude-sonnet-4-20250514") -> Dict[str, Any]:
    """Agentic SMT validation with LLM solver selection"""
    try:
        verifier = AgenticSMTVerifier(anthropic_client, model)
        result = verifier.verify(synthesized_barrier, initial_set, unsafe_set, dynamics)
        return result
        
    except Exception as e:
        print("Agentic verification failed:", str(e))
        return {'success': False, 'error': str(e)}