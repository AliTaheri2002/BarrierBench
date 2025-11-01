import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Union, Any
import re
import logging
import z3
from barrier_parsing import parse_barrier_certificate, clean_and_extract_barrier

logger = logging.getLogger(__name__)



def verify_barrier_conditions_smt(barrier_expr: sp.Expr, variables: List[sp.Symbol], 
                                initial_set: Dict, unsafe_set: Dict, 
                                dynamics: Union[str, Dict, List]) -> Dict[str, Any]:
    # SMT-based verification 

    
    max_retries = 3
    
    for retry_attempt in range(max_retries):
        try:
            logger.info(f"Starting formal SMT verification (attempt {retry_attempt + 1}/{max_retries})...")
            
            z3_vars = {}
            for var in variables:
                z3_vars[str(var)] = z3.Real(str(var))
            
            barrier_z3 = _sympy_to_z3_with_retry(barrier_expr, z3_vars)
            
            results = {
                'condition_1': False,
                'condition_2': False, 
                'condition_3': False,
                'z3_verification': True,
                'verification_details': {}
            }
            
            # Condition 1
            logger.info("Checking Condition 1: B(x) ≤ 0 for all x ∈ X₀")
            cond1_result = _verify_initial_condition_z3_with_retry(barrier_z3, z3_vars, initial_set)
            results['condition_1'] = cond1_result['satisfied']
            results['verification_details']['condition_1'] = cond1_result
            
            # Condition 2
            logger.info("Checking Condition 2: B(x) > 0 for all x ∈ Xᵘ")
            cond2_result = _verify_unsafe_condition_z3_with_retry(barrier_z3, z3_vars, unsafe_set)
            results['condition_2'] = cond2_result['satisfied']
            results['verification_details']['condition_2'] = cond2_result
            
            # Condition 3
            logger.info("Checking Condition 3: ∇B(x)·f(x) < 0 for all x where B(x) = 0")
            cond3_result = _verify_invariance_condition_z3_with_retry(barrier_expr, barrier_z3, z3_vars, dynamics)
            results['condition_3'] = cond3_result['satisfied']
            results['verification_details']['condition_3'] = cond3_result
            
            # Final result
            all_satisfied = all([results['condition_1'], results['condition_2'], results['condition_3']])
            results['all_conditions_satisfied'] = all_satisfied
            
            if all_satisfied:
                logger.info("SUCCESS: ALL 3 CONDITIONS SATISFIED! Valid barrier certificate found!")
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
            logger.warning(f"SMT verification attempt {retry_attempt + 1}/{max_retries} failed: {e}")
            if retry_attempt < max_retries - 1:
                import time
                wait_time = 2 ** retry_attempt
                logger.info(f"Retrying SMT verification in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"SMT verification failed after {max_retries} attempts")
                raise RuntimeError(f"SMT verification failed after {max_retries} retry attempts: {e}")


def _sympy_to_z3_with_retry(sympy_expr: sp.Expr, z3_vars: Dict):
    # convert SymPy expression to Z3
    max_retries = 3
    
    for retry_attempt in range(max_retries):
        try:
            expr_str = str(sympy_expr)
            logger.info(f"Converting expression (attempt {retry_attempt + 1}): {expr_str}")
            
            namespace = {
                **z3_vars,
                '__builtins__': {},
            }
            
            # check all variables x1-x10 are available
            for i in range(1, 11):
                var_name = f'x{i}'
                if var_name not in namespace:
                    namespace[var_name] = z3_vars.get(var_name, z3.Real(var_name))
            
            expr_str = re.sub(r'(\w+)\*\*(\d+)', r'_pow_(\1, \2)', expr_str)
            
            expr_str = re.sub(r'sin\(([^)]+)\)', r'_sin_(\1)', expr_str)
            expr_str = re.sub(r'cos\(([^)]+)\)', r'_cos_(\1)', expr_str)
            expr_str = re.sub(r'exp\(([^)]+)\)', r'_exp_(\1)', expr_str)
            
            # def. pow. func. for Z3 
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
                    # for higher pow.
                    result = base
                    for _ in range(exp - 1):
                        result = result * base
                    return result
            
            # def. trigonometric func. for Z3
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
            
            # def. exp. func. for Z3
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
            logger.warning(f"Z3 conversion attempt {retry_attempt + 1}/{max_retries} failed: {e}")
            if retry_attempt < max_retries - 1:
                continue
            else:
                logger.error(f"Z3 conversion failed after {max_retries} attempts")
                raise RuntimeError(f"Cannot convert expression to Z3 after {max_retries} retry attempts: {e}")


def _verify_initial_condition_z3_with_retry(barrier_z3, z3_vars: Dict, initial_set: Dict) -> Dict[str, Any]:
    # verify B(x) ≤ 0 for all x in initial set using Z3
    max_retries = 3
    
    for retry_attempt in range(max_retries):
        try:
            solver = z3.Solver()
            
            initial_constraints = _get_set_constraints_z3(initial_set, z3_vars)
            for constraint in initial_constraints:
                solver.add(constraint)
            
            solver.add(barrier_z3 > 0)
            
            # check satisfiability with timeout
            solver.set("timeout", 120000)
            result = solver.check()
            
            if result == z3.unsat:
                return {
                    'satisfied': True,
                    'method': 'z3_formal',
                    'details': 'No counterexample found - formally verified'
                }
            elif result == z3.sat:
                model = solver.model()
                counterexample = {str(var): model[z3_var] for var, z3_var in z3_vars.items() if model[z3_var] is not None}
                return {
                    'satisfied': False,
                    'method': 'z3_formal',
                    'details': 'Counterexample found',
                    'counterexample': counterexample
                }
            else:
                if retry_attempt < max_retries - 1:
                    logger.warning(f"Z3 initial condition check returned unknown, retrying...")
                    continue
                else:
                    raise RuntimeError("Z3 solver returned unknown result after all retries")
                    
        except Exception as e:
            logger.warning(f"Z3 initial condition verification attempt {retry_attempt + 1}/{max_retries} failed: {e}")
            if retry_attempt < max_retries - 1:
                continue
            else:
                raise RuntimeError(f"Z3 initial condition verification failed after {max_retries} attempts: {e}")


def _verify_unsafe_condition_z3_with_retry(barrier_z3, z3_vars: Dict, unsafe_set: Dict) -> Dict[str, Any]:
    # verify B(x) > 0 for all x in unsafe set using Z3
    max_retries = 3
    
    for retry_attempt in range(max_retries):
        try:
            solver = z3.Solver()
            
            unsafe_constraints = _get_set_constraints_z3(unsafe_set, z3_vars)
            for constraint in unsafe_constraints:
                solver.add(constraint)
            
            solver.add(barrier_z3 <= 0)
            
            solver.set("timeout", 30000)
            result = solver.check()
            
            if result == z3.unsat:
                return {
                    'satisfied': True,
                    'method': 'z3_formal',
                    'details': 'No counterexample found - formally verified'
                }
            elif result == z3.sat:
                model = solver.model()
                counterexample = {str(var): model[z3_var] for var, z3_var in z3_vars.items() if model[z3_var] is not None}
                return {
                    'satisfied': False,
                    'method': 'z3_formal',
                    'details': 'Counterexample found',
                    'counterexample': counterexample
                }
            else:
                if retry_attempt < max_retries - 1:
                    logger.warning(f"Z3 unsafe condition check returned unknown, retrying...")
                    continue
                else:
                    raise RuntimeError("Z3 solver returned unknown result after all retries")
                    
        except Exception as e:
            logger.warning(f"Z3 unsafe condition verification attempt {retry_attempt + 1}/{max_retries} failed: {e}")
            if retry_attempt < max_retries - 1:
                continue
            else:
                raise RuntimeError(f"Z3 unsafe condition verification failed after {max_retries} attempts: {e}")

def _is_discrete_time(dynamics: Union[str, Dict, List]) -> bool:
    # determine that if dynamics are discrete or continuous-time
    try:
        if isinstance(dynamics, str):
            return '[k+1]' in dynamics or '[k]' in dynamics
        elif isinstance(dynamics, dict):
            first_val = next(iter(dynamics.values())) if dynamics else ""
            return '[k+1]' in str(first_val) or '[k]' in str(first_val)
        elif isinstance(dynamics, list):
            first_val = dynamics[0] if dynamics else ""
            return '[k+1]' in str(first_val) or '[k]' in str(first_val)
        return False
    except Exception:
        return False

def _verify_invariance_condition_z3_with_retry(barrier_expr: sp.Expr, barrier_z3, z3_vars: Dict, 
                                             dynamics: Union[str, Dict, List]) -> Dict[str, Any]:
    # check condition 3, continuous uses ∇B·f < 0, discrete uses B(f(x)) - B(x) ≤ 0
    max_retries = 3    
    is_discrete = _is_discrete_time(dynamics)
    logger.info(f"System type detected: {'DISCRETE-TIME' if is_discrete else 'CONTINUOUS-TIME'}")
    
    for retry_attempt in range(max_retries):
        try:
            barrier_symbols = list(barrier_expr.free_symbols)
            barrier_symbols = sorted(barrier_symbols, key=lambda s: int(str(s)[1:]) if str(s).startswith('x') and str(s)[1:].isdigit() else 999)
            
            all_system_vars = []
            max_var_index = 2 
            
            for sym in barrier_symbols:
                var_str = str(sym)
                if var_str.startswith('x') and var_str[1:].isdigit():
                    var_index = int(var_str[1:])
                    max_var_index = max(max_var_index, var_index)
            
            for i in range(1, max_var_index + 1):
                var_name = f'x{i}'

                existing_sym = None
                for sym in barrier_symbols:
                    if str(sym) == var_name:
                        existing_sym = sym
                        break
                
                if existing_sym:
                    all_system_vars.append(existing_sym)
                else:
                    all_system_vars.append(sp.Symbol(var_name, real=True))
            
            dynamics_exprs = _parse_dynamics(dynamics, all_system_vars)
            if len(dynamics_exprs) != len(all_system_vars):
                raise ValueError(f'Dynamics dimension mismatch: got {len(dynamics_exprs)}, expected {len(all_system_vars)}')
            
            # extract controller parameters from dynamics
            controller_symbols = set()
            for expr in dynamics_exprs:
                expr_symbols = expr.free_symbols
                for sym in expr_symbols:
                    sym_str = str(sym)

                    if sym_str.startswith('u') and (sym_str[1:].isdigit() or len(sym_str) == 1):
                        controller_symbols.add(sym)
            
            logger.info(f"Detected controller parameters: {[str(s) for s in controller_symbols]}")
            
            for var in all_system_vars:
                var_name = str(var)
                if var_name not in z3_vars:
                    z3_vars[var_name] = z3.Real(var_name)
            
            # add controller parameters to z3_vars
            for ctrl_sym in controller_symbols:
                ctrl_name = str(ctrl_sym)
                if ctrl_name not in z3_vars:
                    z3_vars[ctrl_name] = z3.Real(ctrl_name)
                    logger.info(f"Added controller parameter to Z3: {ctrl_name}")
            
            if is_discrete:
                # ========== DISCRETE-TIME: B(f(x)) - B(x) ≤ 0 ==========
                logger.info("Using discrete-time barrier condition: B(f(x)) - B(x) ≤ 0")
                
                # substitute dynamics into barrier to get B(f(x))
                subs_dict = {var: dynamics_exprs[i] for i, var in enumerate(all_system_vars)}
                barrier_next = barrier_expr.subs(subs_dict)
                
                barrier_next_z3 = _sympy_to_z3_with_retry(barrier_next, z3_vars)                
                solver = z3.Solver()
                solver.add(barrier_next_z3 - barrier_z3 > 0)
                
                solver.set("timeout", 30000)
                result = solver.check()
                
                if result == z3.unsat:
                    return {
                        'satisfied': True,
                        'method': 'z3_formal',
                        'details': 'No counterexample found - B(f(x)) - B(x) ≤ 0 formally verified'
                    }
                elif result == z3.sat:
                    model = solver.model()
                    counterexample = {str(var): model[z3_var] for var, z3_var in z3_vars.items() if model[z3_var] is not None}
                    return {
                        'satisfied': False,
                        'method': 'z3_formal',
                        'details': 'Counterexample found - B(f(x)) - B(x) > 0 at some boundary point',
                        'counterexample': counterexample
                    }
                else:
                    if retry_attempt < max_retries - 1:
                        logger.warning(f"Z3 invariance condition check returned unknown, retrying...")
                        continue
                    else:
                        raise RuntimeError("Z3 solver returned unknown result after all retries")
            
            else:
                # ========== CONTINUOUS-TIME: ∇B·f < 0 ==========
                logger.info("using continuous-time barrier condition: ∇B(x)·f(x) < 0")
                
                # compute gradient
                gradient = [sp.diff(barrier_expr, var) for var in all_system_vars]
                
                # validate gradient is not all zeros
                if all(g == 0 for g in gradient):
                    raise ValueError("Gradient is all zeros - barrier may not depend on variables properly")
                
                # compute Lie derivative
                lie_derivative = sum(grad * dyn for grad, dyn in zip(gradient, dynamics_exprs))
                
                # convert to Z3
                lie_derivative_z3 = _sympy_to_z3_with_retry(lie_derivative, z3_vars)
                
                solver = z3.Solver()                
                solver.add(barrier_z3 == 0)
                solver.add(lie_derivative_z3 >= 0)                
                solver.set("timeout", 30000)

                result = solver.check()
                
                if result == z3.unsat:
                    return {
                        'satisfied': True,
                        'method': 'z3_formal',
                        'details': 'No counterexample found - ∇B(x)·f(x) < 0 formally verified'
                    }
                elif result == z3.sat:
                    model = solver.model()
                    counterexample = {str(var): model[z3_var] for var, z3_var in z3_vars.items() if model[z3_var] is not None}
                    return {
                        'satisfied': False,
                        'method': 'z3_formal',
                        'details': 'Counterexample found - ∇B(x)·f(x) ≥ 0 at some boundary point',
                        'counterexample': counterexample
                    }
                else:
                    if retry_attempt < max_retries - 1:
                        logger.warning(f"Z3 invariance condition check returned unknown, retrying...")
                        continue
                    else:
                        raise RuntimeError("Z3 solver returned unknown result after all retries")
                    
        except Exception as e:
            logger.warning(f"Z3 invariance condition verification attempt {retry_attempt + 1}/{max_retries} failed: {e}")
            if retry_attempt < max_retries - 1:
                continue
            else:
                raise RuntimeError(f"Z3 invariance condition verification failed after {max_retries} attempts: {e}")

def _get_set_constraints_z3(set_description: Dict, z3_vars: Dict) -> List:
    constraints = []
    
    try:
        if set_description.get('type') == 'ball':
            radius = set_description.get('radius', 1.0)
            center = set_description.get('center', [0] * len(z3_vars))
            is_complement = set_description.get('complement', False)
            
            if len(center) < len(z3_vars):
                center = center + [0] * (len(z3_vars) - len(center))
            
            squared_distance = 0
            var_list = sorted(z3_vars.items(), key=lambda x: int(x[0][1:]) if x[0].startswith('x') and x[0][1:].isdigit() else 999)
            
            for i, (var_name, z3_var) in enumerate(var_list):
                center_coord = center[i] if i < len(center) else 0
                squared_distance += (z3_var - center_coord) ** 2
            
            if is_complement:
                # outside ball
                constraints.append(squared_distance >= radius ** 2)
            else:
                # inside ball
                constraints.append(squared_distance <= radius ** 2)
        
        elif 'bounds' in set_description:
            bounds = set_description['bounds']
            var_list = sorted(z3_vars.items(), key=lambda x: int(x[0][1:]) if x[0].startswith('x') and x[0][1:].isdigit() else 999)
            
            for i, (var_name, z3_var) in enumerate(var_list):
                if i < len(bounds):
                    low, high = bounds[i]
                    constraints.append(z3_var >= low)
                    constraints.append(z3_var <= high)
        
        else:
            # default
            for var_name, z3_var in z3_vars.items():
                constraints.append(z3_var >= -10)
                constraints.append(z3_var <= 10)
    
    except Exception as e:
        logger.error(f"Error creating Z3 constraints: {e}")

        for var_name, z3_var in z3_vars.items():
            constraints.append(z3_var >= -10)
            constraints.append(z3_var <= 10)
    
    return constraints


def _parse_dynamics(dynamics: Union[str, Dict, List], variables: List[sp.Symbol]) -> List[sp.Expr]:
    # parse dynamics into symbolic expressions
    try:
        if isinstance(dynamics, str):
            return _parse_string_dynamics(dynamics, variables)
        elif isinstance(dynamics, dict):
            return _parse_dict_dynamics(dynamics, variables)
        elif isinstance(dynamics, list):
            return _parse_list_dynamics(dynamics, variables)
        else:
            # def.
            return [-var for var in variables]
            
    except Exception as e:
        logger.error(f"Error parsing dynamics: {e}")
        return [-var for var in variables]


def _parse_string_dynamics(dynamics_str: str, variables: List[sp.Symbol]) -> List[sp.Expr]:
    # parse string dynamics
    try:

        dynamics_str = re.sub(r'\[k\+1\]', '', dynamics_str)                # Remove [k+1]
        dynamics_str = re.sub(r'\[k\]', '', dynamics_str)                   # Remove [k]
        
        equations = [eq.strip() for eq in dynamics_str.split(',')]
        
        exprs = []
        for eq in equations:
            # extract right hand side after "="
            if '=' in eq:
                rhs = eq.split('=')[-1].strip()
            else:
                rhs = eq.strip()
            
            for var in variables:
                var_name = str(var)
                rhs = re.sub(r'\b' + var_name + r'\b', str(var), rhs)
            
            try:
                expr = sp.sympify(rhs)
                exprs.append(expr)
            except Exception as e:
                logger.warning(f"failed to parse dynamics term '{rhs}': {e}")
                # default
                if len(exprs) < len(variables):
                    exprs.append(-variables[len(exprs)])
        
        while len(exprs) < len(variables):
            exprs.append(-variables[len(exprs)])
        
        return exprs[:len(variables)]
        
    except Exception as e:
        logger.error(f"error parsing string dynamics: {e}")
        return [-var for var in variables]


def _parse_dict_dynamics(dynamics_dict: Dict, variables: List[sp.Symbol]) -> List[sp.Expr]:
    exprs = []
    for var in variables:
        var_name = str(var)
        if var_name in dynamics_dict:
            try:
                expr_str = str(dynamics_dict[var_name])
                expr = sp.sympify(expr_str)
                exprs.append(expr)
            except Exception:
                exprs.append(-var) 
        else:
            exprs.append(-var)  
    return exprs


def _parse_list_dynamics(dynamics_list: List, variables: List[sp.Symbol]) -> List[sp.Expr]:
    exprs = []
    for i, expr_item in enumerate(dynamics_list):
        if i >= len(variables):
            break
        try:
            expr = sp.sympify(expr_item)
            exprs.append(expr)
        except Exception:
            exprs.append(-variables[i]) 
    
    while len(exprs) < len(variables):
        exprs.append(-variables[len(exprs)])
    
    return exprs


def validate_barrier_certificate(barrier_str: str, initial_set: Dict, unsafe_set: Dict, 
                               dynamics: Union[str, Dict, List]) -> bool:
    max_retries = 3
    
    for retry_attempt in range(max_retries):
        try:
            barrier_expr, variables = parse_barrier_certificate(barrier_str)
            if barrier_expr is None:
                if retry_attempt < max_retries - 1:
                    logger.warning(f"Barrier parsing failed, retrying... (attempt {retry_attempt + 1}/{max_retries})")
                    continue
                else:
                    logger.error(f"Barrier parsing failed after {max_retries} attempts")
                    return False
            
            results = verify_barrier_conditions_smt(barrier_expr, variables, 
                                                  initial_set, unsafe_set, dynamics)
            return results.get('all_conditions_satisfied', False)
            
        except Exception as e:
            logger.warning(f"Validation attempt {retry_attempt + 1}/{max_retries} failed: {e}")
            if retry_attempt < max_retries - 1:
                import time
                time.sleep(1)
            else:
                logger.error(f"Validation failed after {max_retries} attempts: {e}")
                return False


def get_detailed_condition_results(barrier_str: str, initial_set: Dict, unsafe_set: Dict, 
                                 dynamics: Union[str, Dict, List]) -> Dict[str, Any]:
    # detailed verification results
    max_retries = 3
    
    for retry_attempt in range(max_retries):
        try:
            clean_barrier = clean_and_extract_barrier(barrier_str)
            if not clean_barrier:
                if retry_attempt < max_retries - 1:
                    logger.warning(f"Barrier cleaning failed, retrying... (attempt {retry_attempt + 1}/{max_retries})")
                    continue
                else:
                    return {'success': False, 'error': 'Failed to parse barrier after all retries'}
            
            barrier_expr, variables = parse_barrier_certificate(clean_barrier)
            if barrier_expr is None:
                if retry_attempt < max_retries - 1:
                    logger.warning(f"Barrier expression parsing failed, retrying... (attempt {retry_attempt + 1}/{max_retries})")
                    continue
                else:
                    return {'success': False, 'error': 'Failed to parse barrier expression after all retries'}
            
            conditions = verify_barrier_conditions_smt(barrier_expr, variables, 
                                                     initial_set, unsafe_set, dynamics)
            
            return {
                'success': True,
                'barrier_expression': str(barrier_expr),
                'variables': [str(v) for v in variables],
                'conditions': conditions,
                'all_satisfied': conditions.get('all_conditions_satisfied', False)
            }
            
        except Exception as e:
            logger.warning(f"Detailed results attempt {retry_attempt + 1}/{max_retries} failed: {e}")
            if retry_attempt < max_retries - 1:
                import time
                time.sleep(1)
            else:
                logger.error(f"Detailed results failed after {max_retries} attempts: {e}")
                return {'success': False, 'error': f'All retry attempts failed: {str(e)}'}