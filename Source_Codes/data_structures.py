import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class BarrierSolution:
    """Data class to store barrier certificate solutions"""
    expression: str
    template_type: str
    conditions_satisfied: List[bool]  # [cond1, cond2, cond3]
    verification_details: Dict[str, Any]
    iteration: int
    phase: str  # 'initial', 'refined', 'aggregated'
    score: float  # Number of conditions satisfied
    template_index: int  # Which of the 4 templates this belongs to
    
    def __post_init__(self):
        self.score = sum(self.conditions_satisfied)


@dataclass
class ProblemAnalysis:
    """Data class to store problem analysis results with 4 templates"""
    dynamics_description: str
    set_description: str
    suggested_templates: List[str]  # Now contains 4 templates
    template_reasoning: List[str]   # Reasoning for each template
    mathematical_insights: str
    
    def get_template(self, index: int) -> str:
        """Get template by index with bounds checking"""
        if 0 <= index < len(self.suggested_templates):
            return self.suggested_templates[index]
        raise IndexError(f"Template index {index} out of range. Available templates: {len(self.suggested_templates)}")
    
    def get_reasoning(self, index: int) -> str:
        """Get reasoning by index with bounds checking"""
        if 0 <= index < len(self.template_reasoning):
            return self.template_reasoning[index]
        raise IndexError(f"Reasoning index {index} out of range. Available reasoning: {len(self.template_reasoning)}")


def generate_samples_for_barrier_validation(problem: Dict[str, Any], num_samples: int = 1000) -> Dict[str, Any]:
    """
    Generate samples for barrier validation following k=0 paper methodology
    """
    print(f"DEBUG: Generating {num_samples} samples for barrier validation...")
    
    # Define uncertainty set X (union of initial and unsafe sets with expansion)
    uncertainty_set_X = _define_uncertainty_set_for_validation(problem)
    print(f"DEBUG: Uncertainty set X: {uncertainty_set_X}")
    
    unified_samples = []
    initial_samples_for_compatibility = []
    unsafe_samples_for_compatibility = []
    evolution_samples_for_compatibility = []
    
    # Check if unsafe set is complement
    is_unsafe_complement = problem['unsafe_set'].get('complement', False)
    
    if is_unsafe_complement:
        print(f"DEBUG: Using complement-aware sampling strategy...")
        
        # Strategy 1: Sample from initial set (30%)
        initial_count = int(num_samples * 0.3)
        print(f"DEBUG: Generating {initial_count} samples from initial set...")
        
        for i in range(initial_count):
            try:
                sample_point = _sample_from_set(problem['initial_set'])
                trajectory = _simulate_one_step(sample_point, problem['dynamics'])
                
                is_in_initial = True  # By construction
                is_in_unsafe = _point_in_set(sample_point, problem['unsafe_set'])
                
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
                print(f"ERROR: Failed to generate initial sample {i}: {e}")
                continue
        
        # Strategy 2: Sample from unsafe region (complement) (30%)
        unsafe_count = int(num_samples * 0.3)
        print(f"DEBUG: Generating {unsafe_count} samples from unsafe complement region...")
        
        unsafe_samples_generated = 0
        max_total_attempts = unsafe_count * 20
        
        for attempt in range(max_total_attempts):
            if unsafe_samples_generated >= unsafe_count:
                break
                
            try:
                sample_point = _sample_from_uncertainty_set_X(uncertainty_set_X)
                is_in_unsafe = _point_in_set(sample_point, problem['unsafe_set'])
                
                if is_in_unsafe:
                    trajectory = _simulate_one_step(sample_point, problem['dynamics'])
                    is_in_initial = _point_in_set(sample_point, problem['initial_set'])
                    
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
        
        print(f"DEBUG: Successfully generated {unsafe_samples_generated} unsafe samples")
        
        # Strategy 3: Fill remaining with uniform samples from X
        remaining = num_samples - len(unified_samples)
        print(f"DEBUG: Generating {remaining} uniform samples from uncertainty set X...")
        
        for i in range(remaining):
            try:
                sample_point = _sample_from_uncertainty_set_X(uncertainty_set_X)
                trajectory = _simulate_one_step(sample_point, problem['dynamics'])
                
                is_in_initial = _point_in_set(sample_point, problem['initial_set'])
                is_in_unsafe = _point_in_set(sample_point, problem['unsafe_set'])
                
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
        # Original strategy for non-complement cases
        print(f"DEBUG: Using original strategy for non-complement unsafe set...")
        samples_per_strategy = num_samples // 3
        
        # Strategy 1: Sample from initial set region
        print(f"DEBUG: Generating {samples_per_strategy} samples biased toward initial set...")
        for i in range(samples_per_strategy):
            try:
                sample_point = _sample_from_set(problem['initial_set'])
                trajectory = _simulate_one_step(sample_point, problem['dynamics'])
                
                is_in_initial = _point_in_set(sample_point, problem['initial_set'])
                is_in_unsafe = _point_in_set(sample_point, problem['unsafe_set'])
                
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
        
        # Strategy 2: Sample from unsafe set region
        print(f"DEBUG: Generating {samples_per_strategy} samples biased toward unsafe set...")
        for i in range(samples_per_strategy):
            try:
                sample_point = _sample_from_unsafe_set(problem['unsafe_set'])
                trajectory = _simulate_one_step(sample_point, problem['dynamics'])
                
                is_in_initial = _point_in_set(sample_point, problem['initial_set'])
                is_in_unsafe = _point_in_set(sample_point, problem['unsafe_set'])
                
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
        
        # Strategy 3: Sample uniformly from uncertainty set X
        remaining_samples = num_samples - len(unified_samples)
        print(f"DEBUG: Generating {remaining_samples} samples uniformly from uncertainty set X...")
        for i in range(remaining_samples):
            try:
                sample_point = _sample_from_uncertainty_set_X(uncertainty_set_X)
                trajectory = _simulate_one_step(sample_point, problem['dynamics'])
                
                is_in_initial = _point_in_set(sample_point, problem['initial_set'])
                is_in_unsafe = _point_in_set(sample_point, problem['unsafe_set'])
                
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
    
    # Count final statistics
    initial_count = sum(1 for s in unified_samples if s['in_initial_set'])
    unsafe_count = sum(1 for s in unified_samples if s['in_unsafe_set'])
    
    print(f"DEBUG: Final sample statistics:")
    print(f"DEBUG:   Total samples: {len(unified_samples)}")
    print(f"DEBUG:   In initial set: {initial_count}")
    print(f"DEBUG:   In unsafe set: {unsafe_count}")
    
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




def _define_uncertainty_set_for_validation(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Define uncertainty set X for barrier validation (adapted from k-inductive paper)
    """
    print(f"DEBUG: Defining uncertainty set X for validation...")
    
    initial_set = problem['initial_set']
    unsafe_set = problem['unsafe_set']
    
    print(f"DEBUG: Initial set: {initial_set}")
    print(f"DEBUG: Unsafe set: {unsafe_set}")
    
    if unsafe_set.get('complement', False):
        # For complement unsafe sets: X should be larger than the unsafe boundary
        if unsafe_set.get('type') == 'bounds':
            unsafe_bounds = unsafe_set.get('bounds', [])
            
            # Expand unsafe bounds by 10% to create uncertainty set X
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
            
            print(f"DEBUG: Expanded uncertainty set X for complement: {expanded_bounds}")
            
        elif unsafe_set.get('type') == 'ball':
            center = unsafe_set.get('center', [0, 0])
            radius = unsafe_set.get('radius', 3.0)
            
            expanded_radius = radius * 1.2  # 20% expansion
            
            uncertainty_set = {
                'type': 'ball',
                'center': center,
                'radius': expanded_radius,
                'description': 'uncertainty_set_X_expanded'
            }
            
            print(f"DEBUG: Expanded uncertainty set X for complement ball: radius {expanded_radius}")
        else:
            print(f"ERROR: Unknown unsafe set type: {unsafe_set.get('type')}")
            raise ValueError(f"Unsupported unsafe set type for complement: {unsafe_set.get('type')}")
    
    else:
        # For non-complement cases, create a bounding box that contains both sets
        init_type = initial_set.get('type')
        unsafe_type = unsafe_set.get('type')
        
        # Get bounds for initial set
        if init_type == 'bounds':
            init_bounds = initial_set.get('bounds', [])
        elif init_type == 'ball':
            center = initial_set.get('center', [0, 0])
            radius = initial_set.get('radius', 1.0)
            init_bounds = [[c - radius * 1.1, c + radius * 1.1] for c in center]
        else:
            init_bounds = [[-1, 1], [-1, 1]]  # Default
        
        # Get bounds for unsafe set
        if unsafe_type == 'bounds':
            unsafe_bounds = unsafe_set.get('bounds', [])
        elif unsafe_type == 'ball':
            center = unsafe_set.get('center', [0, 0])
            radius = unsafe_set.get('radius', 1.0)
            unsafe_bounds = [[c - radius * 1.1, c + radius * 1.1] for c in center]
        else:
            unsafe_bounds = [[-1, 1], [-1, 1]]  # Default
        
        # Ensure same dimension
        max_dim = max(len(init_bounds), len(unsafe_bounds))
        while len(init_bounds) < max_dim:
            init_bounds.append([-1, 1])
        while len(unsafe_bounds) < max_dim:
            unsafe_bounds.append([-1, 1])
        
        # Create union bounds
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
        
        print(f"DEBUG: Union uncertainty set X: {union_bounds}")
    
    return uncertainty_set

def _sample_from_set(set_description: Dict[str, Any]) -> List[float]:
    """Sample a point from the given set"""
    try:
        if set_description.get('type') == 'ball':
            center = set_description.get('center', [0, 0])
            radius = set_description.get('radius', 1.0)
            
            dim = len(center)
            random_dir = np.random.randn(dim)
            random_dir = random_dir / np.linalg.norm(random_dir)
            random_radius = radius * (np.random.random() ** (1.0/dim))
            
            sample = np.array(center) + random_radius * random_dir
            return sample.tolist()
            
        elif set_description.get('type') == 'bounds':
            bounds = set_description.get('bounds', [])
            if not bounds:
                return [0.0] * 6  # Default 6D
            
            sample = []
            for bound in bounds:
                if len(bound) >= 2:
                    low, high = bound[0], bound[1]
                    sample.append(np.random.uniform(low, high))
                else:
                    sample.append(0.0)
            
            return sample
        else:
            raise ValueError(f"Unknown set type: {set_description.get('type')}")
            
    except Exception as e:
        print(f"ERROR: Sampling from set failed: {e}")
        return [0.0] * 3  # Default fallback


def _sample_from_unsafe_set(unsafe_set_description: Dict[str, Any]) -> List[float]:
    """Sample a point from unsafe set with guaranteed complement bounds compliance"""
    try:
        if unsafe_set_description.get('type') == 'bounds':
            bounds = unsafe_set_description.get('bounds', [])
            is_complement = unsafe_set_description.get('complement', False)
            
            if not bounds:
                raise ValueError("Empty bounds in unsafe set")
            
            if is_complement:
                # GUARANTEE at least one coordinate is outside bounds
                sample = []
                dim = len(bounds)
                
                violate_dim = np.random.randint(0, dim)
                
                for i, bound in enumerate(bounds):
                    if len(bound) < 2:
                        raise ValueError(f"Invalid bound format: {bound}")
                        
                    low, high = bound[0], bound[1]
                    
                    if i == violate_dim:
                        # GUARANTEE this dimension is outside [low, high]
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
                return _sample_from_set(unsafe_set_description)
                
        elif unsafe_set_description.get('type') == 'ball':
            center = unsafe_set_description.get('center', [0, 0])
            radius = unsafe_set_description.get('radius', 3.0)
            is_complement = unsafe_set_description.get('complement', False)
            
            if is_complement:
                # Sample outside the ball - GUARANTEED
                dim = len(center)
                random_dir = np.random.randn(dim)
                random_dir = random_dir / np.linalg.norm(random_dir)
                
                min_violation_radius = radius * 1.2
                max_violation_radius = radius * 3.0
                random_radius = np.random.uniform(min_violation_radius, max_violation_radius)
                
                sample = np.array(center) + random_radius * random_dir
                return sample.tolist()
            else:
                return _sample_from_set(unsafe_set_description)
        
        else:
            raise ValueError(f"Unknown unsafe set type: {unsafe_set_description.get('type')}")
            
    except Exception as e:
        print(f"ERROR: Unsafe set sampling failed: {e}")
        raise ValueError(f"Failed to sample from unsafe set: {e}")


def _sample_from_uncertainty_set_X(uncertainty_set: Dict[str, Any]) -> List[float]:
    """Sample from the defined uncertainty set X"""
    try:
        return _sample_from_set(uncertainty_set)
    except Exception as e:
        print(f"ERROR: Failed to sample from uncertainty set X: {e}")
        raise


def _point_in_set(point: List[float], set_desc: Dict[str, Any]) -> bool:
    """Check if point is within the given set, handles complement properly"""
    try:
        if set_desc.get('type') == 'ball':
            center = set_desc.get('center', [0, 0])
            radius = set_desc.get('radius', 1.0)
            is_complement = set_desc.get('complement', False)
            
            if len(point) != len(center):
                return False
                
            distance = np.linalg.norm(np.array(point) - np.array(center))
            inside_ball = distance <= radius
            
            return not inside_ball if is_complement else inside_ball
                
        elif set_desc.get('type') == 'bounds':
            bounds = set_desc.get('bounds', [])
            is_complement = set_desc.get('complement', False)
            
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


def _simulate_one_step(initial_point: List[float], dynamics: str) -> List[List[float]]:
    """Simulate exactly 1 step to get [x, f(x)]"""
    try:
        trajectory = [initial_point.copy()]
        next_point = _apply_dynamics_step(initial_point, dynamics)
        trajectory.append(next_point)
        return trajectory
    except Exception as e:
        print(f"ERROR: One-step simulation failed: {e}")
        return [initial_point, initial_point]  # Fallback

def _apply_dynamics_step(point: List[float], dynamics: str) -> List[float]:
    """Apply one dynamics step - simplified version"""
    try:
        
        is_discrete = '[k+1]' in dynamics or '[k]' in dynamics
        
        if is_discrete:
            # DISCRETE: x[k+1] = f(x[k]) - directly use the computed next state
            next_state = _evaluate_dynamics_at_point(point, dynamics)
            return next_state
        else:
            # CONTINUOUS: Simple Euler integration with dt = 0.01
            dt = 0.01
            derivatives = _evaluate_dynamics_at_point(point, dynamics)
            
            next_point = []
            for i, (coord, deriv) in enumerate(zip(point, derivatives)):
                next_coord = coord + dt * deriv
                next_point.append(next_coord)
            
            return next_point
            
    except Exception as e:
        print(f"ERROR: Dynamics step failed: {e}")
        return point  # Fallback to same point

def validate_barrier_on_samples(barrier_expr: str, problem: Dict[str, Any], samples: Dict[str, Any], controller_expr: str = None) -> Dict[str, Any]:
    """
    Validate barrier certificate on samples using k=0 simple validation methodology
    Now supports controller expressions
    """
    try:
        print(f"DEBUG: Simple barrier validation on samples with expression: {barrier_expr}")
        if controller_expr:
            print(f"DEBUG: Controller expression provided: {controller_expr}")
        
        # Handle controller case by substituting into dynamics
        working_problem = problem.copy()
        if controller_expr:
            controller_dict = _parse_controller_expressions_for_samples(controller_expr, problem)
            if controller_dict:
                original_dynamics = problem['dynamics']
                closed_loop_dynamics = _substitute_controller_into_dynamics_for_samples(original_dynamics, controller_dict)
                working_problem['dynamics'] = closed_loop_dynamics
                print(f"DEBUG: Using closed-loop dynamics: {closed_loop_dynamics}")
            else:
                print(f"DEBUG: Failed to parse controller, using original dynamics")
        
        unified_samples = samples['unified_samples']
        barrier_info = _parse_barrier_expression_simple(barrier_expr)
        
        # Check conditions on samples
        condition_1_violations = 0
        condition_2_violations = 0  
        condition_3_violations = 0
        
        condition_1_counterexamples = []
        condition_2_counterexamples = []
        condition_3_counterexamples = []

        tolerance = 1e-6
        
        for sample in unified_samples:
            sample_point = sample['point']
            # Re-simulate trajectory with potentially modified dynamics
            trajectory = _simulate_one_step(sample_point, working_problem['dynamics'])
            
            barrier_value = _evaluate_barrier_simple(barrier_info, sample_point)
            
            # Condition 1: B(x) ≤ 0 for x ∈ X₀
            if sample['in_initial_set']:
                if barrier_value > tolerance:
                    condition_1_violations += 1
                    condition_1_counterexamples.append({
                        'point': sample_point,
                        'barrier_value': barrier_value,
                        'violation': barrier_value,
                    })
            
            # Condition 2: B(x) > 0 for x ∈ Xu
            if sample['in_unsafe_set']:
                if barrier_value <= tolerance:
                    condition_2_violations += 1
                    condition_2_counterexamples.append({
                        'point': sample_point,
                        'barrier_value': barrier_value,
                        'violation': -barrier_value,
                    })
            
            # Condition 3: Check dynamics condition ∇B(x)·f(x) < 0
            if len(trajectory) >= 2:
                x_point = trajectory[0]
                next_point = trajectory[1]
                
                barrier_x = _evaluate_barrier_simple(barrier_info, x_point)
                barrier_next = _evaluate_barrier_simple(barrier_info, next_point)
                
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
        
        # Determine satisfaction
        condition_1_satisfied = condition_1_violations == 0
        condition_2_satisfied = condition_2_violations == 0
        condition_3_satisfied = condition_3_violations == 0
        
        conditions_satisfied = [condition_1_satisfied, condition_2_satisfied, condition_3_satisfied]
        score = sum(conditions_satisfied)
        confidence = 0.99 if score == 3 else (score / 3.0) * 0.8
        
        print(f"DEBUG: Sample-based validation results:")
        print(f"DEBUG:   Condition 1 violations: {condition_1_violations}")
        print(f"DEBUG:   Condition 2 violations: {condition_2_violations}")
        print(f"DEBUG:   Condition 3 violations: {condition_3_violations}")
        print(f"DEBUG:   Score: {score}/3")
        
        return {
            'success': True,
            'conditions_satisfied': conditions_satisfied,
            'score': score,
            'confidence': confidence,
            'counterexamples': {
                'condition_1': condition_1_counterexamples[:5],  # Top 5 worst
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


def _parse_controller_expressions_for_samples(controller_expr: str, problem: Dict[str, Any]) -> Dict[str, str]:
    """Parse controller expression string into dictionary of parameter -> expression mappings"""
    try:
        controller_dict = {}
        controller_params = [p.strip() for p in problem.get('controller_parameters', '').split(',') if p.strip()]
        
        if not controller_params:
            return {}
        
        # Handle comma-separated controller expressions
        expressions = [eq.strip() for eq in controller_expr.split(',') if eq.strip()]
        
        for i, expr in enumerate(expressions):
            expr = expr.strip()
            if '=' in expr:
                # Format: param = expression
                param_name, param_expr = expr.split('=', 1)
                param_name = param_name.strip()
                param_expr = param_expr.strip()
            else:
                # Direct expression, use parameter name
                if i < len(controller_params):
                    param_name = controller_params[i]
                    param_expr = expr
                else:
                    continue
            
            # Clean up the expression
            param_expr = re.sub(r'\s+', ' ', param_expr)
            param_expr = re.sub(r'\s*\+\s*', ' + ', param_expr)
            param_expr = re.sub(r'\s*-\s*', ' - ', param_expr)
            param_expr = re.sub(r'\s*\*\s*', '*', param_expr)
            
            controller_dict[param_name] = param_expr
            
        return controller_dict
        
    except Exception as e:
        print(f"ERROR: Failed to parse controller expressions: {e}")
        return {}


def _substitute_controller_into_dynamics_for_samples(dynamics_str: str, controller_dict: Dict[str, str]) -> str:
    """Substitute controller expressions into dynamics string to create closed-loop dynamics"""
    try:
        if not controller_dict:
            print(f"DEBUG: Empty controller dictionary, returning original dynamics")
            return dynamics_str
            
        # For evaluation purposes, we only need the functional form f(x), not f(x[k])
        dynamics_str = re.sub(r'\[k\+1\]', '', dynamics_str)  # Remove [k+1]
        dynamics_str = re.sub(r'\[k\]', '', dynamics_str)      # Remove [k]

        # Split dynamics into individual equations
        equations = [eq.strip() for eq in dynamics_str.split(',')]
        
        substituted_equations = []
        
        for eq in equations:
            eq = eq.strip()
            
            # Substitute each controller parameter
            for param_name, param_expr in controller_dict.items():
                # Use word boundaries to avoid partial matches
                import re
                pattern = r'\b' + re.escape(param_name) + r'\b'
                
                # Wrap controller expression in parentheses for safety
                replacement = f'({param_expr})'
                
                eq = re.sub(pattern, replacement, eq)
            
            substituted_equations.append(eq)
        
        # Join back into single dynamics string
        closed_loop_dynamics = ', '.join(substituted_equations)
        
        print(f"DEBUG: Original dynamics: {dynamics_str}")
        print(f"DEBUG: Controller: {controller_dict}")
        print(f"DEBUG: Closed-loop dynamics: {closed_loop_dynamics}")
        
        return closed_loop_dynamics
        
    except Exception as e:
        print(f"ERROR: Failed to substitute controller into dynamics: {e}")
        return dynamics_str  # Return original as fallback


def _evaluate_dynamics_at_point(point: List[float], dynamics: str) -> List[float]:
    """Evaluate dynamics at a given point - simplified version with controller support"""
    try:
        # print(dynamics)
        # exit()
        is_discrete = '[k+1]' in dynamics or '[k]' in dynamics
        
        # For discrete systems, strip the [k] notation for evaluation
        if is_discrete:
            dynamics = re.sub(r'\[k\+1\]', '', dynamics)  # Remove [k+1]
            dynamics = re.sub(r'\[k\]', '', dynamics) 

        # Parse dynamics string and evaluate
        equations = [eq.strip() for eq in dynamics.split(',')]
        derivatives = []
        
        # Create variable mapping for up to 10 variables
        var_map = {f'x{i+1}': point[i] if i < len(point) else 0.0 for i in range(10)}
        
        for eq in equations:
            if '=' in eq:
                rhs = eq.split('=')[1].strip()
                # Simple evaluation using eval with safe context
                safe_dict = var_map.copy()
                safe_dict.update({
                    '__builtins__': {},
                    'exp': math.exp,
                    'sin': math.sin,
                    'cos': math.cos,
                    'sqrt': math.sqrt,
                    'abs': abs,
                    'max': max,
                    'min': min,
                    'pow': pow,
                    'tanh': math.tanh
                })
                
                try:
                    result = eval(rhs, safe_dict)
                    derivatives.append(float(result))
                except Exception as eval_error:
                    print(f"DEBUG: Eval failed for '{rhs}': {eval_error}")
                    derivatives.append(0.0)  # Fallback
        
        # Pad with zeros if needed
        while len(derivatives) < len(point):
            derivatives.append(0.0)
            
        return derivatives
    except Exception as e:
        print(f"ERROR: Dynamics evaluation failed: {e}")
        return [0.0] * len(point)

def _parse_barrier_expression_simple(barrier_expr: str) -> Dict[str, Any]:
    """Simple barrier parsing for sample validation"""
    return {
        'expression': barrier_expr,
        'original': barrier_expr
    }


def _evaluate_barrier_simple(barrier_info: Dict[str, Any], point: List[float]) -> float:
    """Evaluate barrier function at given point - simplified version"""
    try:
        expression = barrier_info['expression']
        
        # Create variable mapping
        var_map = {f'x{i+1}': point[i] if i < len(point) else 0.0 for i in range(10)}
        
        safe_dict = var_map.copy()
        safe_dict.update({
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
        
        result = eval(expression, safe_dict)
        
        if not math.isfinite(result):
            return 0.0
            
        return float(result)
        
    except Exception as e:
        print(f"ERROR: Barrier evaluation failed: {e}")
        return 0.0