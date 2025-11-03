import numpy as np
import math
import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def generate_samples_for_barrier_validation(problem: Dict[str, Any], num_samples: int = 1000) -> Dict[str, Any]:
    # Generate samples
    print(f"Generating {num_samples} samples for barrier validation") 
    state_space_X = compute_state_space_bounds(problem)
    print(f"State space X: {state_space_X}")
    
    unified_samples = []
    initial_samples_for_compatibility = []
    unsafe_samples_for_compatibility = []
    evolution_samples_for_compatibility = []
    
    # check if unsafe set is complement
    is_unsafe_complement = problem['unsafe_set'].get('complement', False)
    
    if is_unsafe_complement:
        print(f"Using complement-aware sampling strategy")
        # 1. sample from initial set
        initial_count = int(num_samples * 0.3)
        print(f"Generating {initial_count} samples from initial set")
        
        for i in range(initial_count):
            try:
                sample_point = sample_from_set(problem['initial_set'])
                trajectory = simulate_one_step(sample_point, problem['dynamics'])
                
                is_in_initial = True  
                is_in_unsafe = is_point_in_set(sample_point, problem['unsafe_set'])
                
                sample_data = {
                    'point': sample_point,
                    'trajectory': trajectory,
                    'in_initial_set': is_in_initial,
                    'in_unsafe_set': is_in_unsafe,
                    'strategy': 'initial_targeted'
                }
                
                unified_samples.append(sample_data)
                initial_samples_for_compatibility.append({'point': sample_point, 'trajectory': trajectory})
                
            except Exception as e:
                print(f"ERROR: failed to generate initial sample {i}: {e}")
                continue
        
        # 2. sample from unsafe region
        unsafe_count = int(num_samples * 0.3)
        print(f"Generating {unsafe_count} samples from unsafe complement region")
        
        unsafe_samples_generated = 0
        max_total_attempts = unsafe_count * 20
        
        for _ in range(max_total_attempts):
            if unsafe_samples_generated >= unsafe_count:
                break
                
            try:
                sample_point = sample_from_set(state_space_X)
                is_in_unsafe = is_point_in_set(sample_point, problem['unsafe_set'])
                
                if is_in_unsafe:
                    trajectory = simulate_one_step(sample_point, problem['dynamics'])
                    is_in_initial = is_point_in_set(sample_point, problem['initial_set'])
                    
                    sample_data = {
                        'point': sample_point,
                        'trajectory': trajectory,
                        'in_initial_set': is_in_initial,
                        'in_unsafe_set': is_in_unsafe,
                        'strategy': 'unsafe_targeted'
                    }
                    
                    unified_samples.append(sample_data)
                    unsafe_samples_for_compatibility.append({'point': sample_point})
                    unsafe_samples_generated += 1
                    
            except Exception as e:
                continue
        
        print(f"Successfully generated {unsafe_samples_generated} unsafe samples")
        
        # 3 
        num_remaining = num_samples - len(unified_samples)
        print(f"Generating {num_remaining} uniform samples from state space set X")
        
        for i in range(num_remaining):
            try:
                sample_point = sample_from_set(state_space_X)
                trajectory = simulate_one_step(sample_point, problem['dynamics'])
                
                is_in_initial = is_point_in_set(sample_point, problem['initial_set'])
                is_in_unsafe = is_point_in_set(sample_point, problem['unsafe_set'])
                
                sample_data = {
                    'point': sample_point,
                    'trajectory': trajectory,
                    'in_initial_set': is_in_initial,
                    'in_unsafe_set': is_in_unsafe,
                    'strategy': 'uniform'
                }
                
                unified_samples.append(sample_data)
                evolution_samples_for_compatibility.append({'point': sample_point, 'trajectory': trajectory})
                
            except Exception as e:
                print(f"ERROR: Failed to generate uniform sample {i}: {e}")
                continue
    
    else:
        print(f"using original strategy for non-complement unsafe set")
        samples_per_strategy = num_samples // 3
        
        # 1.
        print(f"Generating {samples_per_strategy} samples biased toward initial set")
        for i in range(samples_per_strategy):
            try:
                sample_point = sample_from_set(problem['initial_set'])
                trajectory = simulate_one_step(sample_point, problem['dynamics'])
                
                is_in_initial = is_point_in_set(sample_point, problem['initial_set'])
                is_in_unsafe = is_point_in_set(sample_point, problem['unsafe_set'])
                
                sample_data = {
                    'point': sample_point,
                    'trajectory': trajectory,
                    'in_initial_set': is_in_initial,
                    'in_unsafe_set': is_in_unsafe,
                    'strategy': 'initial_biased'
                }
                
                unified_samples.append(sample_data)
                initial_samples_for_compatibility.append({'point': sample_point, 'trajectory': trajectory})
                
            except Exception as e:
                print(f"ERROR: Failed to generate initial-biased sample {i}: {e}")
                continue
        
        # 2.
        print(f"Generating {samples_per_strategy} samples biased toward unsafe set")
        for i in range(samples_per_strategy):
            try:
                sample_point = sample_from_unsafe_set(problem['unsafe_set'])
                trajectory = simulate_one_step(sample_point, problem['dynamics'])
                
                is_in_initial = is_point_in_set(sample_point, problem['initial_set'])
                is_in_unsafe = is_point_in_set(sample_point, problem['unsafe_set'])
                
                sample_data = {
                    'point': sample_point,
                    'trajectory': trajectory,
                    'in_initial_set': is_in_initial,
                    'in_unsafe_set': is_in_unsafe,
                    'strategy': 'unsafe_biased'
                }
                
                unified_samples.append(sample_data)
                unsafe_samples_for_compatibility.append({'point': sample_point})
                
            except Exception as e:
                print(f"ERROR: Failed to generate unsafe-biased sample {i}: {e}")
                continue
        
        # 3.
        remaining_samples = num_samples - len(unified_samples)
        print(f"Generating {remaining_samples} samples uniformly from state space set X")
        for i in range(remaining_samples):
            try:
                sample_point = sample_from_set(state_space_X)
                trajectory = simulate_one_step(sample_point, problem['dynamics'])
                
                is_in_initial = is_point_in_set(sample_point, problem['initial_set'])
                is_in_unsafe = is_point_in_set(sample_point, problem['unsafe_set'])
                
                sample_data = {
                    'point': sample_point,
                    'trajectory': trajectory,
                    'in_initial_set': is_in_initial,
                    'in_unsafe_set': is_in_unsafe,
                    'strategy': 'uniform'
                }
                
                unified_samples.append(sample_data)
                evolution_samples_for_compatibility.append({'point': sample_point, 'trajectory': trajectory})
                
            except Exception as e:
                print(f"ERROR: Failed to generate uniform sample {i}: {e}")
                continue
    
    # count statistics
    initial_count = sum(1 for s in unified_samples if s['in_initial_set'])
    unsafe_count = sum(1 for s in unified_samples if s['in_unsafe_set'])
    
    print(f"sample statistics:")
    print(f"Total samples: {len(unified_samples)}")
    print(f"In initial set: {initial_count}")
    print(f"In unsafe set: {unsafe_count}")
    
    return {
        'unified_samples': unified_samples,
        'initial': initial_samples_for_compatibility,
        'unsafe': unsafe_samples_for_compatibility,
        'evolution': evolution_samples_for_compatibility,
        'statistics': {
            'total_samples': len(unified_samples),
            'initial_set_count': initial_count,
            'unsafe_set_count': unsafe_count
        }
    }

def compute_state_space_bounds(problem: Dict[str, Any]) -> Dict[str, Any]:
    initial_set = problem['initial_set']
    unsafe_set = problem['unsafe_set']
    
    if unsafe_set.get('complement', False):
        # for complement unsafe sets, X should be larger than the unsafe boundary
        if unsafe_set.get('type') == 'bounds':
            unsafe_bounds = unsafe_set.get('bounds', [])
            
            # expand unsafe bounds by 10% to create state space set X
            expanded_bounds = []
            for bound_pair in unsafe_bounds:
                if len(bound_pair) >= 2:
                    low, high = bound_pair[0], bound_pair[1]
                    center = (low + high) / 2
                    half_width = (high - low) / 2
                    
                    expanded_half_width = half_width * 1.1  # 10% expansion
                    expanded_low = center - expanded_half_width
                    expanded_high = center + expanded_half_width
                    
                    expanded_bounds.append([expanded_low, expanded_high])
                else:
                    expanded_bounds.append(bound_pair)
            
            uncertainty_set = {
                'type': 'bounds',
                'bounds': expanded_bounds,
                'description': 'uncertainty_set_X_expanded'
            }
            
            print(f"Expanded state space set X for complement: {expanded_bounds}")
            
        elif unsafe_set.get('type') == 'ball':
            center = unsafe_set.get('center')
            radius = unsafe_set.get('radius')
            
            expanded_radius = radius * 1.2  # 20% expansion
            
            uncertainty_set = {
                'type': 'ball',
                'center': center,
                'radius': expanded_radius,
                'description': 'uncertainty_set_X_expanded'
            }
            
            print(f"Expanded state space set X for complement ball: radius {expanded_radius}")
        else:
            print(f"ERROR: Unknown unsafe set type: {unsafe_set.get('type')}")
            raise ValueError(f"Unsupported unsafe set type for complement: {unsafe_set.get('type')}")
    
    else:
        init_type = initial_set.get('type')
        unsafe_type = unsafe_set.get('type')
        
        # get bounds for initial set
        if init_type == 'bounds':
            init_bounds = initial_set.get('bounds')
        elif init_type == 'ball':
            center = initial_set.get('center')
            radius = initial_set.get('radius')
            init_bounds = [[c - radius * 1.1, c + radius * 1.1] for c in center]
        else:
            init_bounds = [[-1, 1], [-1, 1]]  # default
        
        # get bounds for unsafe set
        if unsafe_type == 'bounds':
            unsafe_bounds = unsafe_set.get('bounds')
        elif unsafe_type == 'ball':
            center = unsafe_set.get('center')
            radius = unsafe_set.get('radius')
            unsafe_bounds = [[c - radius * 1.1, c + radius * 1.1] for c in center]
        else:
            logger.error(f"Unknown unsafe set type: '{unsafe_type}'")
            return None
        
        # create union bounds
        union_bounds = []
        for init_bound, unsafe_bound in zip(init_bounds, unsafe_bounds):
            union_low = min(init_bound[0], unsafe_bound[0])
            union_high = max(init_bound[1], unsafe_bound[1])
            union_bounds.append([union_low, union_high])
        
        uncertainty_set = {
            'type': 'bounds',
            'bounds': union_bounds,
            'description': 'uncertainty_set_X_union'
        }
        
        print(f"union state space set X: {union_bounds}")
    
    return uncertainty_set

def sample_from_set(set_description: Dict[str, Any]) -> List[float]:
    try:
        if set_description.get('type') == 'ball':
            center = set_description.get('center')
            radius = set_description.get('radius')
            
            dim = len(center)
            random_dir = np.random.randn(dim)
            random_dir = random_dir / np.linalg.norm(random_dir)
            random_radius = radius * (np.random.random() ** (1.0/dim))
            
            sample = np.array(center) + random_radius * random_dir
            return sample.tolist()
            
        elif set_description.get('type') == 'bounds':
            bounds = set_description.get('bounds', [])
            if bounds is None or not bounds:
                logger.error("Bounds set description missing or empty 'bounds' field")
                return None
            
            sample = []
            for i, bound in enumerate(bounds):
                if len(bound) >= 2:
                    low, high = bound[0], bound[1]
                    sample.append(np.random.uniform(low, high))
                else:
                    logger.error(f"Invalid bound at index {i}: {bound}")
                    return None
            
            return sample
        else:
            raise ValueError(f"Unknown set type: {set_description.get('type')}")
            
    except Exception as e:
        print(f"ERROR: Sampling from set failed: {e}")
        return None

def sample_from_unsafe_set(unsafe_set_description: Dict[str, Any]) -> List[float]:
    try:
        if unsafe_set_description.get('type') == 'bounds':
            bounds = unsafe_set_description.get('bounds')
            is_complement = unsafe_set_description.get('complement', False)
            
            if not bounds:
                raise ValueError("Empty bounds in unsafe set")
            
            if is_complement:
                sample = []
                dim = len(bounds)
                
                violate_dim = np.random.randint(0, dim)
                
                for i, bound in enumerate(bounds):
                    if len(bound) < 2:
                        raise ValueError(f"Invalid bound format: {bound}")
                        
                    low, high = bound[0], bound[1]
                    
                    if i == violate_dim:
                        margin = max(abs(low), abs(high)) * 0.5 + 1.0
                        if np.random.random() < 0.5:
                            violation_val = np.random.uniform(low - margin, low - 0.1)
                        else:
                            violation_val = np.random.uniform(high + 0.1, high + margin)
                        
                        sample.append(violation_val)
                    else:
                        sample.append(np.random.uniform(low, high))
                
                return sample
            else:
                return sample_from_set(unsafe_set_description)
                
        elif unsafe_set_description.get('type') == 'ball':
            center = unsafe_set_description.get('center')
            radius = unsafe_set_description.get('radius')
            is_complement = unsafe_set_description.get('complement', False)
            
            if is_complement:
                dim = len(center)
                random_dir = np.random.randn(dim)
                random_dir = random_dir / np.linalg.norm(random_dir)
                
                min_violation_radius = radius * 1.2
                max_violation_radius = radius * 3.0
                random_radius = np.random.uniform(min_violation_radius, max_violation_radius)
                
                sample = np.array(center) + random_radius * random_dir
                return sample.tolist()
            else:
                return sample_from_set(unsafe_set_description)
        
        else:
            raise ValueError(f"Unknown unsafe set type: {unsafe_set_description.get('type')}")
            
    except Exception as e:
        print(f"ERROR: Unsafe set sampling failed: {e}")
        raise ValueError(f"Failed to sample from unsafe set: {e}")

def is_point_in_set(point: List[float], set_description: Dict[str, Any]) -> bool:
    try:
        if set_description.get('type') == 'ball':
            center = set_description.get('center')
            radius = set_description.get('radius')
            is_complement = set_description.get('complement', False)
            
            if len(point) != len(center):
                return False
                
            distance = np.linalg.norm(np.array(point) - np.array(center))
            inside_ball = distance <= radius
            
            return not inside_ball if is_complement else inside_ball
                
        elif set_description.get('type') == 'bounds':
            bounds = set_description.get('bounds')
            is_complement = set_description.get('complement', False)
            
            if len(point) != len(bounds):
                return False
                
            inside_bounds = True
            for i, (coord, bound) in enumerate(zip(point, bounds)):
                if len(bound) < 2:
                    return False
                if coord < bound[0] or coord > bound[1]:
                    inside_bounds = False
                    break
            
            return not inside_bounds if is_complement else inside_bounds

        else:
            return False
                
    except Exception as e:
        print(f"ERROR: Point-in-set check failed: {e}")
        return False

def simulate_one_step(initial_point: List[float], dynamics: str) -> List[List[float]]:
    # simulate 1 step

    if isinstance(dynamics, str):
            is_discrete = '[k+1]' in dynamics or '[k]' in dynamics
    else:
        raise TypeError(f"Invalid dynamics type - expected str")
    
    try:
        trajectory = [initial_point.copy()]

        if is_discrete:
            next_point = dynamics_function(initial_point, dynamics)
        else:
            dt = 0.01
            derivatives = dynamics_function(initial_point, dynamics)
            next_point = []
            for i, (coord, deriv) in enumerate(zip(initial_point, derivatives)):
                next_coord = coord + dt * deriv
                next_point.append(next_coord)
            
        trajectory.append(next_point)

        return trajectory

    except Exception as e:
        print(f"ERROR: One-step simulation failed: {e}")
        return None

def validate_barrier_on_samples(barrier_expr: str, problem: Dict[str, Any], samples: Dict[str, Any], controller_expr: str = None) -> Dict[str, Any]:
    # validate barrier certificate on samples
    try:
        print(f"Simple barrier validation on samples with expression: {barrier_expr}")
        if controller_expr:
            print(f"Controller expression provided: {controller_expr}")
        
        # handle controller
        working_problem = problem.copy()
        if controller_expr:
            controller_dict = parse_controller_expressions(controller_expr, problem)
            if controller_dict:
                original_dynamics = problem['dynamics']
                closed_loop_dynamics = substitute_controller_into_dynamics_for_samples(original_dynamics, controller_dict)
                working_problem['dynamics'] = closed_loop_dynamics
                print(f"Using closed-loop dynamics: {closed_loop_dynamics}")
            else:
                print(f"Failed to parse controller, using original dynamics")
        
        unified_samples = samples['unified_samples']
        
        # Check conditions on samples
        condition_1_violations = condition_2_violations = condition_3_violations = 0
        
        condition_1_counterexamples = []
        condition_2_counterexamples = []
        condition_3_counterexamples = []

        tolerance = 1e-6
        
        for sample in unified_samples:
            sample_point = sample['point']

            trajectory = simulate_one_step(sample_point, working_problem['dynamics'])            
            barrier_value = barrier_function(barrier_expr, sample_point)
            
            # Condition 1
            if sample['in_initial_set']:
                if barrier_value > tolerance:
                    condition_1_violations += 1
                    condition_1_counterexamples.append({
                        'point': sample_point,
                        'barrier_value': barrier_value,
                        'violation': barrier_value,
                    })
            
            # Condition 2
            if sample['in_unsafe_set']:
                if barrier_value <= tolerance:
                    condition_2_violations += 1
                    condition_2_counterexamples.append({
                        'point': sample_point,
                        'barrier_value': barrier_value,
                        'violation': -barrier_value,
                    })
            
            # Condition 3
            if trajectory is not None:

                x_point = trajectory[0]
                next_point = trajectory[1]
                
                barrier_x = barrier_function(barrier_expr, x_point)
                barrier_next = barrier_function(barrier_expr, next_point)
                
                # For standard barriers: B(f(x)) - B(x) ≤ 0
                dynamics_violation = barrier_next - barrier_x
                if dynamics_violation > tolerance:
                    condition_3_violations += 1
                    condition_3_counterexamples.append({
                        'trajectory': trajectory,
                        'barrier_x': barrier_x,
                        'barrier_next': barrier_next,
                        'violation': dynamics_violation,
                    })
            else:
                raise RuntimeError("Trajectory generation failed, cannot proceed with barrier validation")

        # Determine satisfaction
        condition_1_satisfied = condition_1_violations == 0
        condition_2_satisfied = condition_2_violations == 0
        condition_3_satisfied = condition_3_violations == 0
        
        conditions_satisfied = [condition_1_satisfied, condition_2_satisfied, condition_3_satisfied]
        score = sum(conditions_satisfied)
        confidence = 0.99 if score == 3 else (score / 3.0) * 0.8
        
        print(f"Sample-based validation results:")
        print(f"Condition 1 violations: {condition_1_violations}")
        print(f"Condition 2 violations: {condition_2_violations}")
        print(f"Condition 3 violations: {condition_3_violations}")
        print(f"Score: {score}/3")
        
        return {
            'success': True,
            'conditions_satisfied': conditions_satisfied,
            'score': score,
            'confidence': confidence,
            'counterexamples': {                                        # this only write if you want to see or debug, we don't use it in our paper
                'condition_1': condition_1_counterexamples[:5], 
                'condition_2': condition_2_counterexamples[:5],
                'condition_3': condition_3_counterexamples[:5]
            },
            'violation_counts': [condition_1_violations, condition_2_violations, condition_3_violations]
        }
        
    except Exception as e:
        print(f"ERROR: Sample-based validation failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def parse_controller_expressions(controller_expr: str, problem: Dict[str, Any]) -> Dict[str, str]:
    try:
        controller_dict = {}
        controller_params = [p.strip() for p in problem.get('controller_parameters', '').split(',') if p.strip()]
        
        if not controller_params:
            return {}
        
        # handle comma-separated controller expressions
        expressions = [eq.strip() for eq in controller_expr.split(',') if eq.strip()]
        
        for i, expr in enumerate(expressions):
            expr = expr.strip()
            if '=' in expr:
                # format" param = expression
                param_name, param_expr = expr.split('=', 1)
                param_name = param_name.strip()
                param_expr = param_expr.strip()
            else:
                if i < len(controller_params):
                    param_name = controller_params[i]
                    param_expr = expr
                else:
                    print(f"ERROR: Extra controller expression at position {i+1} with no matching parameter")
                    continue
            
            param_expr = re.sub(r'\s+', ' ', param_expr)
            param_expr = re.sub(r'\s*\+\s*', ' + ', param_expr)
            param_expr = re.sub(r'\s*-\s*', ' - ', param_expr)
            param_expr = re.sub(r'\s*\*\s*', '*', param_expr)
            
            controller_dict[param_name] = param_expr
            
        return controller_dict
        
    except Exception as e:
        print(f"ERROR: Failed to parse controller expressions: {e}")
        return {}

def substitute_controller_into_dynamics_for_samples(dynamics_str: str, controller_dict: Dict[str, str]) -> str:
    # substitute controller expressions into dynamics string to create closed-loop dynamics
    try:
        if not controller_dict:
            print(f"Empty controller dictionary, returning original dynamics")
            return dynamics_str
            
        dynamics_str = re.sub(r'\[k\+1\]', '', dynamics_str)                # Remove [k+1]
        dynamics_str = re.sub(r'\[k\]', '', dynamics_str)                   # Remove [k]

        system_equations = [eq.strip() for eq in dynamics_str.split(',')]
        
        substituted_equations = []
        
        for eq in system_equations:
            eq = eq.strip()
            
            for param_name, param_expr in controller_dict.items():
                pattern = r'\b' + re.escape(param_name) + r'\b'
                replacement = f'({param_expr})'
                eq = re.sub(pattern, replacement, eq)
            
            substituted_equations.append(eq)
        
        closed_loop_dynamics = ', '.join(substituted_equations)
        
        print(f"Original dynamics: {dynamics_str}")
        print(f"Controller: {controller_dict}")
        print(f"Closed-loop dynamics: {closed_loop_dynamics}")
        
        return closed_loop_dynamics
        
    except Exception as e:
        print(f"ERROR: Failed to substitute controller into dynamics: {e}")
        return None
        # return dynamics_str 

def dynamics_function(point: List[float], dynamics: str) -> List[float]:
    # evaluate dynamics at a given point
    try:
        if isinstance(dynamics, str):
            is_discrete = '[k+1]' in dynamics or '[k]' in dynamics
        else:
            raise TypeError(f"Invalid dynamics type - expected str")
        
        if is_discrete:
            dynamics = re.sub(r'\[k\+1\]', '', dynamics)                    # Remove [k+1]
            dynamics = re.sub(r'\[k\]', '', dynamics) 

        system_equations = [eq.strip() for eq in dynamics.split(',')]
        derivatives = []
        
        variable_map = {f'x{i+1}': point[i] if i < len(point) else 0.0 for i in range(10)}
        
        for eq in system_equations:
            if '=' in eq:
                rhs = eq.split('=')[1].strip()
        
                var_values = variable_map.copy()
                var_values.update({
                    '__builtins__': {},
                    'exp':  math.exp,
                    'sin':  math.sin, 'cos': math.cos,
                    'sqrt': math.sqrt, 'abs': abs,
                    'max':  max, 'min': min,
                    'pow':  pow, 'tanh': math.tanh})
                
                result = eval(rhs, var_values)
                derivatives.append(float(result))
        
        if len(derivatives) != len(system_equations):
            logger.error(f"Dimension mismatch: expected {len(system_equations)} derivatives, got {len(derivatives)}")
            return None
            
        return derivatives
    
    except Exception as e:
        print(f"ERROR: Dynamics evaluation failed: {e}")
        return None

def barrier_function(expression: str, point: List[float]) -> float:
    # evaluate barrier function at given point
    try:
        
        variable_map = {f'x{i+1}': point[i] if i < len(point) else 0.0 for i in range(10)}
        
        var_values = variable_map.copy()
        var_values.update({
            '__builtins__': {},
            'exp': math.exp,
            'sin': math.sin,
            'cos': math.cos,
            'sqrt': math.sqrt,
            'abs': abs,
            'max': max,
            'min': min,
            'pow': pow
        })
        
        result = eval(expression, var_values)
        
        if not math.isfinite(result):
            print(f"ERROR: Barrier evaluation resulted in non-finite value ({result})")
            return None
            
        return float(result)
        
    except Exception as e:
        print(f"ERROR: Barrier evaluation failed: {e}")
        return None