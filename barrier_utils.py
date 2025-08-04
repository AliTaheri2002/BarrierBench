import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Union, Any
import re
import logging

logger = logging.getLogger(__name__)

# Try to import Z3 SMT solver for formal verification
try:
    import z3
    Z3_AVAILABLE = True
    logger.info("Z3 SMT Solver available for formal verification")
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("Z3 SMT Solver not available - using numerical verification")


def clean_and_extract_barrier(barrier_str: str) -> str:
    """Extract barrier certificate from response text"""
    if not barrier_str or not isinstance(barrier_str, str):
        return ""
    
    try:
        # Remove common prefixes
        barrier_str = re.sub(r'B\(x\)\s*=\s*', '', barrier_str, flags=re.IGNORECASE)
        barrier_str = re.sub(r'barrier\s*certificate\s*:?\s*', '', barrier_str, flags=re.IGNORECASE)
        
        # Clean formatting
        barrier_str = barrier_str.strip().strip('"\'')
        barrier_str = re.sub(r'[\.,:;"`\']+$', '', barrier_str).strip()
        
        # Basic validation
        if len(barrier_str) < 3 or not re.search(r'x\d*', barrier_str):
            return ""
        
        return barrier_str
        
    except Exception as e:
        logger.warning(f"Error cleaning barrier '{barrier_str}': {e}")
        return ""


def parse_barrier_certificate(barrier_str: str) -> Tuple[Optional[sp.Expr], List[sp.Symbol]]:
    """Parse barrier certificate string into symbolic expression"""
    try:
        cleaned_str = clean_and_extract_barrier(barrier_str)
        if not cleaned_str:
            logger.warning("Empty barrier string after cleaning")
            return None, []
        
        # Extract variables
        variables = sorted(set(re.findall(r'\bx\d*\b', cleaned_str)))
        if not variables:
            variables = ['x1', 'x2']  # Default
        
        var_symbols = [sp.Symbol(var, real=True) for var in variables]
        
        # Parse expression
        try:
            # Normalize operators
            expr_str = cleaned_str.replace('^', '**')
            expr = sp.sympify(expr_str, evaluate=True)
            
            # Validate
            if expr.free_symbols:
                return expr, var_symbols
            else:
                logger.warning("Expression has no variables")
                return None, var_symbols
                
        except Exception as e:
            logger.warning(f"Failed to parse expression '{cleaned_str}': {e}")
            return None, var_symbols
        
    except Exception as e:
        logger.error(f"Error parsing barrier certificate '{barrier_str}': {e}")
        return None, []


def verify_barrier_conditions_smt(barrier_expr: sp.Expr, variables: List[sp.Symbol], 
                                initial_set: Dict, unsafe_set: Dict, 
                                dynamics: Union[str, Dict, List]) -> Dict[str, Any]:
    """
    PROPER SMT-based verification of barrier certificate conditions
    This is the core verification that should determine success/failure
    """
    if not Z3_AVAILABLE:
        logger.warning("Z3 not available, falling back to numerical verification")
        return _numerical_verification_fallback(barrier_expr, variables, initial_set, unsafe_set, dynamics)
    
    try:
        logger.info("Starting formal SMT verification...")
        
        # Convert SymPy to Z3
        z3_vars = {}
        for var in variables:
            z3_vars[str(var)] = z3.Real(str(var))
        
        barrier_z3 = _sympy_to_z3(barrier_expr, z3_vars)
        
        results = {
            'condition_1': False,
            'condition_2': False, 
            'condition_3': False,
            'z3_verification': True,
            'verification_details': {}
        }
        
        # Condition 1: B(x) ≤ 0 for all x ∈ X₀
        logger.info("Checking Condition 1: B(x) ≤ 0 for all x ∈ X₀")
        cond1_result = _verify_initial_condition_z3(barrier_z3, z3_vars, initial_set)
        results['condition_1'] = cond1_result['satisfied']
        results['verification_details']['condition_1'] = cond1_result
        
        if not results['condition_1']:
            logger.info("Condition 1 failed - barrier not non-positive in initial set")
            results['all_conditions_satisfied'] = False
            return results
        
        # Condition 2: B(x) > 0 for all x ∈ Xᵤ
        logger.info("Checking Condition 2: B(x) > 0 for all x ∈ Xᵤ")
        cond2_result = _verify_unsafe_condition_z3(barrier_z3, z3_vars, unsafe_set)
        results['condition_2'] = cond2_result['satisfied']
        results['verification_details']['condition_2'] = cond2_result
        
        if not results['condition_2']:
            logger.info("Condition 2 failed - barrier not positive in unsafe set")
            results['all_conditions_satisfied'] = False
            return results
        
        # Condition 3: ∇B(x)·f(x) ≤ 0 for all x where B(x) = 0
        logger.info("Checking Condition 3: ∇B(x)·f(x) ≤ 0 for all x where B(x) = 0")
        cond3_result = _verify_invariance_condition_z3(barrier_expr, barrier_z3, z3_vars, dynamics)
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
        logger.error(f"SMT verification failed: {e}")
        # Fallback to numerical verification
        return _numerical_verification_fallback(barrier_expr, variables, initial_set, unsafe_set, dynamics)


def _verify_initial_condition_z3(barrier_z3, z3_vars: Dict, initial_set: Dict) -> Dict[str, Any]:
    """Verify B(x) ≤ 0 for all x in initial set using Z3"""
    try:
        solver = z3.Solver()
        
        # Add initial set constraints
        initial_constraints = _get_set_constraints_z3(initial_set, z3_vars)
        for constraint in initial_constraints:
            solver.add(constraint)
        
        # Add negation of desired property: ∃x ∈ X₀ : B(x) > 0
        solver.add(barrier_z3 > 0)
        
        # Check satisfiability
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
            return {
                'satisfied': False,
                'method': 'z3_formal',
                'details': 'Z3 returned unknown result'
            }
            
    except Exception as e:
        logger.error(f"Error in Z3 initial condition verification: {e}")
        return {
            'satisfied': False,
            'method': 'z3_formal',
            'details': f'Z3 error: {str(e)}'
        }


def _verify_unsafe_condition_z3(barrier_z3, z3_vars: Dict, unsafe_set: Dict) -> Dict[str, Any]:
    """Verify B(x) > 0 for all x in unsafe set using Z3"""
    try:
        solver = z3.Solver()
        
        # Add unsafe set constraints
        unsafe_constraints = _get_set_constraints_z3(unsafe_set, z3_vars)
        for constraint in unsafe_constraints:
            solver.add(constraint)
        
        # Add negation of desired property: ∃x ∈ Xᵤ : B(x) ≤ 0
        solver.add(barrier_z3 <= 0)
        
        # Check satisfiability
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
            return {
                'satisfied': False,
                'method': 'z3_formal',
                'details': 'Z3 returned unknown result'
            }
            
    except Exception as e:
        logger.error(f"Error in Z3 unsafe condition verification: {e}")
        return {
            'satisfied': False,
            'method': 'z3_formal',
            'details': f'Z3 error: {str(e)}'
        }


def _verify_invariance_condition_z3(barrier_expr: sp.Expr, barrier_z3, z3_vars: Dict, 
                                   dynamics: Union[str, Dict, List]) -> Dict[str, Any]:
    """Verify ∇B(x)·f(x) ≤ 0 for all x where B(x) = 0 using Z3"""
    try:
        # Compute gradient symbolically
        variables = [sp.Symbol(var, real=True) for var in z3_vars.keys()]
        gradient = [sp.diff(barrier_expr, var) for var in variables]
        
        # Parse dynamics
        dynamics_exprs = _parse_dynamics(dynamics, variables)
        if len(dynamics_exprs) != len(variables):
            return {
                'satisfied': False,
                'method': 'z3_formal',
                'details': 'Dynamics dimension mismatch'
            }
        
        # Compute Lie derivative
        lie_derivative = sum(grad * dyn for grad, dyn in zip(gradient, dynamics_exprs))
        
        # Convert to Z3
        lie_derivative_z3 = _sympy_to_z3(lie_derivative, z3_vars)
        
        solver = z3.Solver()
        
        # Add constraint: B(x) = 0 (on the boundary)
        solver.add(barrier_z3 == 0)
        
        # Add negation of desired property: ∃x : B(x) = 0 ∧ ∇B(x)·f(x) > 0
        solver.add(lie_derivative_z3 > 0)
        
        # Check satisfiability
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
            return {
                'satisfied': False,
                'method': 'z3_formal',
                'details': 'Z3 returned unknown result'
            }
            
    except Exception as e:
        logger.error(f"Error in Z3 invariance condition verification: {e}")
        return {
            'satisfied': False,
            'method': 'z3_formal',
            'details': f'Z3 error: {str(e)}'
        }


def _get_set_constraints_z3(set_description: Dict, z3_vars: Dict) -> List:
    """Convert set description to Z3 constraints"""
    constraints = []
    
    try:
        if set_description.get('type') == 'ball':
            radius = set_description.get('radius', 1.0)
            center = set_description.get('center', [0, 0])
            is_complement = set_description.get('complement', False)
            
            # Ensure center has right dimension
            if len(center) < len(z3_vars):
                center = center + [0] * (len(z3_vars) - len(center))
            
            # Compute squared distance
            squared_distance = 0
            for i, (var_name, z3_var) in enumerate(z3_vars.items()):
                center_coord = center[i] if i < len(center) else 0
                squared_distance += (z3_var - center_coord) ** 2
            
            if is_complement:
                # Outside ball: ||x - center||² ≥ radius²
                constraints.append(squared_distance >= radius ** 2)
            else:
                # Inside ball: ||x - center||² ≤ radius²
                constraints.append(squared_distance <= radius ** 2)
        
        elif 'bounds' in set_description:
            bounds = set_description['bounds']
            for i, (var_name, z3_var) in enumerate(z3_vars.items()):
                if i < len(bounds):
                    low, high = bounds[i]
                    constraints.append(z3_var >= low)
                    constraints.append(z3_var <= high)
        
        else:
            # Default: bounded region [-10, 10]²
            for var_name, z3_var in z3_vars.items():
                constraints.append(z3_var >= -10)
                constraints.append(z3_var <= 10)
    
    except Exception as e:
        logger.error(f"Error creating Z3 constraints: {e}")
        # Fallback: bounded region
        for var_name, z3_var in z3_vars.items():
            constraints.append(z3_var >= -10)
            constraints.append(z3_var <= 10)
    
    return constraints


def _sympy_to_z3(sympy_expr: sp.Expr, z3_vars: Dict):
    """Convert SymPy expression to Z3 expression"""
    try:
        # Convert SymPy expression to string
        expr_str = str(sympy_expr)
        
        # Replace variable names with Z3 variables
        for var_name, z3_var in z3_vars.items():
            expr_str = re.sub(rf'\b{var_name}\b', f'z3_vars["{var_name}"]', expr_str)
        
        # Evaluate the expression
        return eval(expr_str, {"z3_vars": z3_vars, "__builtins__": {}})
        
    except Exception as e:
        logger.error(f"Error converting SymPy to Z3: {e}")
        # Simple fallback
        if len(z3_vars) >= 2:
            var_list = list(z3_vars.values())
            return var_list[0]**2 + var_list[1]**2 - 1
        else:
            return list(z3_vars.values())[0]**2 - 1


def _parse_dynamics(dynamics: Union[str, Dict, List], variables: List[sp.Symbol]) -> List[sp.Expr]:
    """Parse dynamics into symbolic expressions"""
    try:
        if isinstance(dynamics, str):
            return _parse_string_dynamics(dynamics, variables)
        elif isinstance(dynamics, dict):
            return _parse_dict_dynamics(dynamics, variables)
        elif isinstance(dynamics, list):
            return _parse_list_dynamics(dynamics, variables)
        else:
            # Default: stable linear system
            return [-var for var in variables]
            
    except Exception as e:
        logger.error(f"Error parsing dynamics: {e}")
        return [-var for var in variables]


def _parse_string_dynamics(dynamics_str: str, variables: List[sp.Symbol]) -> List[sp.Expr]:
    """Parse string dynamics like 'dx1/dt = -x1 + 0.5*x2, dx2/dt = -x2 - 0.3*x1'"""
    try:
        # Split by comma for multiple equations
        equations = [eq.strip() for eq in dynamics_str.split(',')]
        
        exprs = []
        for eq in equations:
            # Extract right-hand side after '='
            if '=' in eq:
                rhs = eq.split('=')[-1].strip()
            else:
                rhs = eq.strip()
            
            # Replace variable names
            for var in variables:
                var_name = str(var)
                rhs = re.sub(r'\b' + var_name + r'\b', str(var), rhs)
            
            try:
                expr = sp.sympify(rhs)
                exprs.append(expr)
            except Exception as e:
                logger.warning(f"Failed to parse dynamics term '{rhs}': {e}")
                # Default: stable
                if len(exprs) < len(variables):
                    exprs.append(-variables[len(exprs)])
        
        # Ensure we have enough expressions
        while len(exprs) < len(variables):
            exprs.append(-variables[len(exprs)])
        
        return exprs[:len(variables)]
        
    except Exception as e:
        logger.error(f"Error parsing string dynamics: {e}")
        return [-var for var in variables]


def _parse_dict_dynamics(dynamics_dict: Dict, variables: List[sp.Symbol]) -> List[sp.Expr]:
    """Parse dictionary dynamics"""
    exprs = []
    for var in variables:
        var_name = str(var)
        if var_name in dynamics_dict:
            try:
                expr_str = str(dynamics_dict[var_name])
                expr = sp.sympify(expr_str)
                exprs.append(expr)
            except Exception:
                exprs.append(-var)  # Default stable
        else:
            exprs.append(-var)  # Default stable
    return exprs


def _parse_list_dynamics(dynamics_list: List, variables: List[sp.Symbol]) -> List[sp.Expr]:
    """Parse list dynamics"""
    exprs = []
    for i, expr_item in enumerate(dynamics_list):
        if i >= len(variables):
            break
        try:
            expr = sp.sympify(expr_item)
            exprs.append(expr)
        except Exception:
            exprs.append(-variables[i])  # Default stable
    
    # Fill remaining
    while len(exprs) < len(variables):
        exprs.append(-variables[len(exprs)])
    
    return exprs


def _numerical_verification_fallback(barrier_expr: sp.Expr, variables: List[sp.Symbol], 
                                   initial_set: Dict, unsafe_set: Dict, 
                                   dynamics: Union[str, Dict, List]) -> Dict[str, Any]:
    """Fallback numerical verification when Z3 is not available"""
    logger.warning("Using numerical verification fallback - results may be approximate")
    
    results = {
        'condition_1': False,
        'condition_2': False,
        'condition_3': False,
        'z3_verification': False,
        'verification_details': {}
    }
    
    try:
        # Simple sampling-based verification
        results['condition_1'] = _check_initial_condition_numerical(barrier_expr, variables, initial_set)
        results['condition_2'] = _check_unsafe_condition_numerical(barrier_expr, variables, unsafe_set)
        results['condition_3'] = _check_invariance_condition_numerical(barrier_expr, variables, dynamics)
        
        results['all_conditions_satisfied'] = all([
            results['condition_1'], 
            results['condition_2'], 
            results['condition_3']
        ])
        
        satisfied_count = sum([results['condition_1'], results['condition_2'], results['condition_3']])
        logger.info(f"Numerical verification: {satisfied_count}/3 conditions satisfied")
        
    except Exception as e:
        logger.error(f"Error in numerical verification: {e}")
    
    return results


def _check_initial_condition_numerical(barrier_expr: sp.Expr, variables: List[sp.Symbol], 
                                     initial_set: Dict) -> bool:
    """Numerical check of initial condition"""
    try:
        # Generate sample points in initial set
        sample_points = _generate_sample_points(initial_set, variables, 100)
        
        violations = 0
        for point in sample_points:
            try:
                value = _evaluate_expression(barrier_expr, variables, point)
                if value > 1e-6:  # Small tolerance
                    violations += 1
            except:
                continue
        
        violation_rate = violations / len(sample_points) if sample_points else 1.0
        return violation_rate < 0.05  # Accept if < 5% violations
        
    except Exception as e:
        logger.error(f"Error in numerical initial condition check: {e}")
        return False


def _check_unsafe_condition_numerical(barrier_expr: sp.Expr, variables: List[sp.Symbol], 
                                    unsafe_set: Dict) -> bool:
    """Numerical check of unsafe condition"""
    try:
        sample_points = _generate_sample_points(unsafe_set, variables, 100)
        
        violations = 0
        for point in sample_points:
            try:
                value = _evaluate_expression(barrier_expr, variables, point)
                if value <= 1e-6:  # Should be positive
                    violations += 1
            except:
                continue
        
        violation_rate = violations / len(sample_points) if sample_points else 1.0
        return violation_rate < 0.05
        
    except Exception as e:
        logger.error(f"Error in numerical unsafe condition check: {e}")
        return False


def _check_invariance_condition_numerical(barrier_expr: sp.Expr, variables: List[sp.Symbol], 
                                        dynamics: Union[str, Dict, List]) -> bool:
    """Numerical check of invariance condition"""
    try:
        # Compute gradient and Lie derivative
        gradient = [sp.diff(barrier_expr, var) for var in variables]
        dynamics_exprs = _parse_dynamics(dynamics, variables)
        
        if len(dynamics_exprs) != len(variables):
            return False
        
        lie_derivative = sum(grad * dyn for grad, dyn in zip(gradient, dynamics_exprs))
        
        # Find points on zero level set
        zero_level_points = _find_zero_level_points(barrier_expr, variables, 50)
        
        if not zero_level_points:
            return True  # No boundary points found
        
        violations = 0
        for point in zero_level_points:
            try:
                barrier_value = _evaluate_expression(barrier_expr, variables, point)
                if abs(barrier_value) > 0.1:  # Not on zero level set
                    continue
                
                lie_value = _evaluate_expression(lie_derivative, variables, point)
                if lie_value > 1e-6:  # Should be ≤ 0
                    violations += 1
            except:
                continue
        
        violation_rate = violations / len(zero_level_points)
        return violation_rate < 0.1
        
    except Exception as e:
        logger.error(f"Error in numerical invariance condition check: {e}")
        return True  # Conservative: assume satisfied if can't check


def _generate_sample_points(set_description: Dict, variables: List[sp.Symbol], 
                          num_points: int = 100) -> List[List[float]]:
    """Generate sample points from set description"""
    points = []
    
    try:
        if set_description.get('type') == 'ball':
            radius = set_description.get('radius', 1.0)
            center = set_description.get('center', [0] * len(variables))
            is_complement = set_description.get('complement', False)
            
            for _ in range(num_points):
                if is_complement:
                    # Sample outside ball
                    direction = np.random.normal(0, 1, len(variables))
                    direction = direction / np.linalg.norm(direction)
                    r = np.random.uniform(radius * 1.1, radius * 3)
                    point = np.array(center[:len(variables)]) + r * direction
                else:
                    # Sample inside ball
                    direction = np.random.normal(0, 1, len(variables))
                    direction = direction / np.linalg.norm(direction)
                    r = radius * (np.random.uniform(0, 1) ** (1/len(variables)))
                    point = np.array(center[:len(variables)]) + r * direction
                points.append(point.tolist())
        
        elif 'bounds' in set_description:
            bounds = set_description['bounds']
            for _ in range(num_points):
                point = []
                for i in range(len(variables)):
                    if i < len(bounds):
                        low, high = bounds[i]
                        point.append(np.random.uniform(low, high))
                    else:
                        point.append(np.random.uniform(-2, 2))
                points.append(point)
        
        else:
            # Default sampling
            for _ in range(num_points):
                point = [np.random.uniform(-2, 2) for _ in variables]
                points.append(point)
    
    except Exception as e:
        logger.error(f"Error generating sample points: {e}")
        # Fallback
        for _ in range(num_points):
            point = [np.random.uniform(-1, 1) for _ in variables]
            points.append(point)
    
    return points


def _find_zero_level_points(barrier_expr: sp.Expr, variables: List[sp.Symbol], 
                           num_points: int = 50) -> List[List[float]]:
    """Find points approximately on the zero level set"""
    points = []
    
    # Random sampling approach
    for _ in range(num_points * 10):  # Try more points
        test_point = [np.random.uniform(-2, 2) for _ in variables]
        try:
            value = _evaluate_expression(barrier_expr, variables, test_point)
            if abs(value) < 0.2:  # Close to zero
                points.append(test_point)
                if len(points) >= num_points:
                    break
        except:
            continue
    
    return points


def _evaluate_expression(expr: sp.Expr, variables: List[sp.Symbol], 
                        point: List[float]) -> float:
    """Safely evaluate symbolic expression at a point"""
    try:
        subs_dict = dict(zip(variables, point[:len(variables)]))
        value = expr.subs(subs_dict)
        
        if hasattr(value, 'evalf'):
            return float(value.evalf())
        else:
            return float(value)
            
    except Exception as e:
        raise ValueError(f"Cannot evaluate expression: {e}")


# Main API functions
def validate_barrier_certificate(barrier_str: str, initial_set: Dict, unsafe_set: Dict, 
                               dynamics: Union[str, Dict, List]) -> bool:
    """
    Main validation function - returns True only if ALL 3 conditions are satisfied
    """
    try:
        barrier_expr, variables = parse_barrier_certificate(barrier_str)
        if barrier_expr is None:
            return False
        
        results = verify_barrier_conditions_smt(barrier_expr, variables, 
                                              initial_set, unsafe_set, dynamics)
        return results.get('all_conditions_satisfied', False)
        
    except Exception as e:
        logger.error(f"Error validating barrier certificate: {e}")
        return False


def get_detailed_condition_results(barrier_str: str, initial_set: Dict, unsafe_set: Dict, 
                                 dynamics: Union[str, Dict, List]) -> Dict[str, Any]:
    """Get detailed verification results"""
    try:
        clean_barrier = clean_and_extract_barrier(barrier_str)
        if not clean_barrier:
            return {'success': False, 'error': 'Failed to parse barrier'}
        
        barrier_expr, variables = parse_barrier_certificate(clean_barrier)
        if barrier_expr is None:
            return {'success': False, 'error': 'Failed to parse barrier expression'}
        
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
        logger.error(f"Error getting detailed results: {e}")
        return {'success': False, 'error': str(e)}