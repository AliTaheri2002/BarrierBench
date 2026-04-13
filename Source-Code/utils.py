import numpy as np
import math
import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def generate_samples_for_barrier_validation(problem, num_samples=1000):
    state_space_X = compute_state_space_bounds(problem)
    unified_samples, initial_samples, unsafe_samples, evolution_samples = [], [], [], []

    is_unsafe_complement = problem['unsafe_set'].get('complement', False)

    if is_unsafe_complement:
        initial_count = int(num_samples * 0.3)
        for _ in range(initial_count):
            point = sample_from_set(problem['initial_set'])
            if point is None:
                continue
            trajectory = simulate_one_step(point, problem['dynamics'])
            sample = {'point': point, 'trajectory': trajectory, 'in_initial_set': True,
                      'in_unsafe_set': is_point_in_set(point, problem['unsafe_set']),
                      'strategy': 'initial_targeted'}
            unified_samples.append(sample)
            initial_samples.append({'point': point, 'trajectory': trajectory})

        unsafe_count = int(num_samples * 0.3)
        unsafe_generated = 0
        for _ in range(unsafe_count * 20):
            if unsafe_generated >= unsafe_count:
                break
            point = sample_from_set(state_space_X)
            if point is None or not is_point_in_set(point, problem['unsafe_set']):
                continue
            trajectory = simulate_one_step(point, problem['dynamics'])
            unified_samples.append({
                'point': point, 'trajectory': trajectory,
                'in_initial_set': is_point_in_set(point, problem['initial_set']),
                'in_unsafe_set': True, 'strategy': 'unsafe_targeted'})
            unsafe_samples.append({'point': point})
            unsafe_generated += 1

        for _ in range(num_samples - len(unified_samples)):
            point = sample_from_set(state_space_X)
            if point is None:
                continue
            trajectory = simulate_one_step(point, problem['dynamics'])
            unified_samples.append({
                'point': point, 'trajectory': trajectory,
                'in_initial_set': is_point_in_set(point, problem['initial_set']),
                'in_unsafe_set': is_point_in_set(point, problem['unsafe_set']), 'strategy': 'uniform'})
            evolution_samples.append({'point': point, 'trajectory': trajectory})

    else:
        samples_per_strategy = num_samples // 3

        for _ in range(samples_per_strategy):
            point = sample_from_set(problem['initial_set'])
            if point is None:
                continue
            trajectory = simulate_one_step(point, problem['dynamics'])
            unified_samples.append({
                'point': point, 'trajectory': trajectory,
                'in_initial_set': is_point_in_set(point, problem['initial_set']),
                'in_unsafe_set': is_point_in_set(point, problem['unsafe_set']), 'strategy': 'initial_biased'
            })
            initial_samples.append({'point': point, 'trajectory': trajectory})

        for _ in range(samples_per_strategy):
            point = sample_from_unsafe_set(problem['unsafe_set'])
            if point is None:
                continue
            trajectory = simulate_one_step(point, problem['dynamics'])
            unified_samples.append({
                'point': point, 'trajectory': trajectory,
                'in_initial_set': is_point_in_set(point, problem['initial_set']),
                'in_unsafe_set': is_point_in_set(point, problem['unsafe_set']), 'strategy': 'unsafe_biased'
            })
            unsafe_samples.append({'point': point})

        for _ in range(num_samples - len(unified_samples)):
            point = sample_from_set(state_space_X)
            if point is None:
                continue
            trajectory = simulate_one_step(point, problem['dynamics'])
            unified_samples.append({
                'point': point, 'trajectory': trajectory,
                'in_initial_set': is_point_in_set(point, problem['initial_set']),
                'in_unsafe_set': is_point_in_set(point, problem['unsafe_set']), 'strategy': 'uniform'
            })
            evolution_samples.append({'point': point, 'trajectory': trajectory})

    initial_count = sum(1 for s in unified_samples if s['in_initial_set'])
    unsafe_count = sum(1 for s in unified_samples if s['in_unsafe_set'])

    return {
        'unified_samples': unified_samples, 'initial': initial_samples,
        'unsafe': unsafe_samples, 'evolution': evolution_samples,
        'statistics': {'total_samples': len(unified_samples), 'initial_set_count': initial_count,
        'unsafe_set_count': unsafe_count}
    }


def compute_state_space_bounds(problem):
    initial_set = problem['initial_set']
    unsafe_set = problem['unsafe_set']

    if unsafe_set.get('complement', False):
        unsafe_type = unsafe_set.get('type')

        if unsafe_type in ('bounds', 'box'):
            expanded_bounds = []
            for low, high in unsafe_set.get('bounds', []):
                center = (low + high) / 2
                half_width = (high - low) / 2 * 1.1
                expanded_bounds.append([center - half_width, center + half_width])
            return {'type': 'bounds', 'bounds': expanded_bounds}

        elif unsafe_type == 'ball':
            return {
                'type': 'ball',
                'center': unsafe_set.get('center'),
                'radius': unsafe_set.get('radius') * 1.2
            }
        else:
            raise ValueError(f"Unsupported unsafe set type for complement: {unsafe_type}")

    # non-complement--union of initial and unsafe bounds
    def get_bounds(s):
        t = s.get('type')
        if t in ('bounds', 'box'):
            return s.get('bounds')
        elif t == 'ball':
            c, r = s.get('center'), s.get('radius')
            return [[ci - r * 1.1, ci + r * 1.1] for ci in c]
        elif t == 'union':
            sets = s.get('sets', [])
            if not sets:
                return None
            all_bounds = [get_bounds(sub) for sub in sets if get_bounds(sub)]
            if not all_bounds:
                return None
            merged = list(all_bounds[0])
            for b in all_bounds[1:]:
                for i, (mb, sb) in enumerate(zip(merged, b)):
                    merged[i] = [min(mb[0], sb[0]), max(mb[1], sb[1])]
            return merged
        return None

    init_bounds = get_bounds(initial_set) or [[-1, 1], [-1, 1]]
    unsafe_bounds = get_bounds(unsafe_set)

    if not unsafe_bounds:
        logger.error(f"Unknown unsafe set type: '{unsafe_set.get('type')}'")
        return None

    union_bounds = [
        [min(ib[0], ub[0]), max(ib[1], ub[1])]
        for ib, ub in zip(init_bounds, unsafe_bounds)
    ]
    return {'type': 'bounds', 'bounds': union_bounds}


def sample_from_set(set_description):
    set_type = set_description.get('type')

    if set_type == 'ball':
        center = set_description.get('center')
        radius = set_description.get('radius')
        if center is None or radius is None:
            logger.error("Ball set missing 'center' or 'radius'")
            return None
        dim = len(center)
        d = np.random.randn(dim)
        d /= np.linalg.norm(d)
        r = radius * (np.random.random() ** (1.0 / dim))
        return (np.array(center) + r * d).tolist()

    elif set_type in ('bounds', 'box'):
        bounds = set_description.get('bounds', [])
        if not bounds:
            logger.error("Bounds set missing 'bounds'")
            return None
        return [np.random.uniform(low, high) for low, high in bounds]

    elif set_type == 'union':
        sets = set_description.get('sets', [])
        if not sets:
            logger.error("Union set missing 'sets'")
            return None
        return sample_from_set(sets[np.random.randint(0, len(sets))])

    else:
        logger.error(f"Unknown set type: {set_type}")
        return None


def sample_from_unsafe_set(unsafe_set_description):
    set_type = unsafe_set_description.get('type')

    if set_type == 'union':
        sets = unsafe_set_description.get('sets', [])
        if not sets:
            raise ValueError("Empty sets in union")
        return sample_from_unsafe_set(sets[np.random.randint(0, len(sets))])

    if set_type in ('bounds', 'box'):
        bounds = unsafe_set_description.get('bounds')
        is_complement = unsafe_set_description.get('complement', False)
        if not bounds:
            raise ValueError("Empty bounds in unsafe set")

        if is_complement:
            dim = len(bounds)
            violate_dim = np.random.randint(0, dim)
            sample = []
            for i, (low, high) in enumerate(bounds):
                if i == violate_dim:
                    margin = max(abs(low), abs(high)) * 0.5 + 1.0
                    if np.random.random() < 0.5:
                        sample.append(np.random.uniform(low - margin, low - 0.1))
                    else:
                        sample.append(np.random.uniform(high + 0.1, high + margin))
                else:
                    sample.append(np.random.uniform(low, high))
            return sample
        else:
            return sample_from_set(unsafe_set_description)

    elif set_type == 'ball':
        center = unsafe_set_description.get('center')
        radius = unsafe_set_description.get('radius')
        is_complement = unsafe_set_description.get('complement', False)

        if is_complement:
            dim = len(center)
            d = np.random.randn(dim)
            d /= np.linalg.norm(d)
            r = np.random.uniform(radius * 1.2, radius * 3.0)
            return (np.array(center) + r * d).tolist()
        else:
            return sample_from_set(unsafe_set_description)

    else:
        raise ValueError(f"Unknown unsafe set type: {set_type}")


def is_point_in_set(point, set_description):
    if point is None:
        return False

    set_type = set_description.get('type')

    if set_type == 'union':
        return any(is_point_in_set(point, s) for s in set_description.get('sets', []))

    if set_type == 'ball':
        center = set_description.get('center')
        radius = set_description.get('radius')
        is_complement = set_description.get('complement', False)
        if center is None or radius is None:
            return False
        inside = np.linalg.norm(np.array(point) - np.array(center)) <= radius
        return not inside if is_complement else inside

    elif set_type in ('bounds', 'box'):
        bounds = set_description.get('bounds')
        is_complement = set_description.get('complement', False)
        if bounds is None:
            return False
        inside = all(bound[0] <= coord <= bound[1] for coord, bound in zip(point, bounds))
        return not inside if is_complement else inside

    logger.error(f"Unknown set type: {set_type}")
    return False


def simulate_one_step(initial_point, dynamics):
    if not initial_point:
        return None

    if not isinstance(dynamics, str):
        raise TypeError("Invalid dynamics type - expected str")

    is_discrete = '[k+1]' in dynamics or '[k]' in dynamics
    trajectory = [initial_point.copy()]

    if is_discrete:
        next_point = dynamics_function(initial_point, dynamics)
    else:
        dt = 0.01
        derivatives = dynamics_function(initial_point, dynamics)
        if derivatives is None:
            return None
        next_point = [coord + dt * deriv for coord, deriv in zip(initial_point, derivatives)]

    if next_point is None:
        return None

    trajectory.append(next_point)
    return trajectory


def validate_barrier_on_samples(barrier_expr, problem, samples, controller_expr=None):

    working_problem = problem.copy()
    if controller_expr:
        controller_dict = parse_controller_expressions(controller_expr, problem)
        if controller_dict:
            closed_loop = substitute_controller_into_dynamics_for_samples(problem['dynamics'], controller_dict)
            working_problem['dynamics'] = closed_loop

    unified_samples = samples['unified_samples']
    c1_violations = c2_violations = c3_violations = 0
    c1_counterexamples = c2_counterexamples = c3_counterexamples = []
    tolerance = 1e-6

    for sample in unified_samples:
        point = sample['point']
        trajectory = simulate_one_step(point, working_problem['dynamics'])
        barrier_value = barrier_function(barrier_expr, point)

        if sample['in_initial_set'] and barrier_value > tolerance:
            c1_violations += 1
            c1_counterexamples.append({'point': point, 'barrier_value': barrier_value})

        if sample['in_unsafe_set'] and barrier_value <= tolerance:
            c2_violations += 1
            c2_counterexamples.append({'point': point, 'barrier_value': barrier_value})

        if trajectory is not None:
            bx = barrier_function(barrier_expr, trajectory[0])
            bn = barrier_function(barrier_expr, trajectory[1])
            if bn - bx > tolerance:
                c3_violations += 1
                c3_counterexamples.append({'trajectory': trajectory, 'barrier_x': bx, 'barrier_next': bn})
        else:
            raise RuntimeError("Trajectory generation failed")

    c1_ok = c1_violations == 0
    c2_ok = c2_violations == 0
    c3_ok = c3_violations == 0
    score = sum([c1_ok, c2_ok, c3_ok])

    return {
        'success': True,
        'conditions_satisfied': [c1_ok, c2_ok, c3_ok],
        'score': score,
        'confidence': 0.99 if score == 3 else (score / 3.0),
        'condition_details': {'condition_1_failed_count': c1_violations, 'condition_2_failed_count': c2_violations,
        'condition_3_failed_count': c3_violations},
        'counterexamples': {'condition_1': c1_counterexamples[:5], 'condition_2': c2_counterexamples[:5], 'condition_3': c3_counterexamples[:5]},
        'violation_counts': [c1_violations, c2_violations, c3_violations]
    }


def parse_controller_expressions(controller_expr, problem):
    controller_dict = {}
    controller_params = [p.strip() for p in problem.get('controller_parameters', '').split(',') if p.strip()]

    if not controller_params:
        return {}

    expressions = [eq.strip() for eq in controller_expr.split(',') if eq.strip()]

    for i, expr in enumerate(expressions):
        if '=' in expr:
            param_name, param_expr = expr.split('=', 1)
            param_name = param_name.strip()
            param_expr = param_expr.strip()
        elif i < len(controller_params):
            param_name = controller_params[i]
            param_expr = expr
        else:
            logger.warning(f"Extra controller expression at position {i+1}")
            continue

        param_expr = re.sub(r'\s+', ' ', param_expr)
        param_expr = re.sub(r'\s*\+\s*', ' + ', param_expr)
        param_expr = re.sub(r'\s*-\s*', ' - ', param_expr)
        param_expr = re.sub(r'\s*\*\s*', '*', param_expr)
        controller_dict[param_name] = param_expr

    return controller_dict


def substitute_controller_into_dynamics_for_samples(dynamics_str, controller_dict):
    if not controller_dict:
        return dynamics_str

    dynamics_str = re.sub(r'\[k\+1\]', '', dynamics_str)
    dynamics_str = re.sub(r'\[k\]', '', dynamics_str)

    equations = [eq.strip() for eq in dynamics_str.split(',')]
    substituted = []

    for eq in equations:
        for param, expr in controller_dict.items():
            eq = re.sub(r'\b' + re.escape(param) + r'\b', f'({expr})', eq)
        substituted.append(eq)

    return ', '.join(substituted)


def dynamics_function(point, dynamics):
    if isinstance(dynamics, str):
        is_discrete = '[k+1]' in dynamics or '[k]' in dynamics
    else:
        raise TypeError("Invalid dynamics type - expected str")

    if is_discrete:
        dynamics = re.sub(r'\[k\+1\]', '', dynamics)
        dynamics = re.sub(r'\[k\]', '', dynamics)

    system_equations = [eq.strip() for eq in dynamics.split(',')]
    variable_map = {f'x{i+1}': point[i] if i < len(point) else 0.0 for i in range(10)}
    for i in range(10):
        variable_map[f'u{i}'] = 0.0
    variable_map['u'] = 0.0

    derivatives = []
    for eq in system_equations:
        if '=' in eq:
            rhs = eq.split('=')[1].strip()
            var_values = {**variable_map, '__builtins__': {}, 'exp': math.exp, 'sin': math.sin, 'cos': math.cos,
                          'sqrt': math.sqrt, 'abs': abs, 'max': max, 'min': min, 'pow': pow, 'tanh': math.tanh}
            result = eval(rhs, var_values)
            derivatives.append(float(result))

    if len(derivatives) != len(system_equations):
        logger.error(f"Dimension mismatch: {len(derivatives)} vs {len(system_equations)}")
        return None

    return derivatives


def barrier_function(expression, point):
    if not point:
        return None

    variable_map = {f'x{i+1}': point[i] if i < len(point) else 0.0 for i in range(10)}
    var_values = {**variable_map, '__builtins__': {}, 'exp': math.exp, 'sin': math.sin, 'cos': math.cos,
                  'sqrt': math.sqrt, 'abs': abs, 'max': max, 'min': min, 'pow': pow}

    result = eval(expression, var_values)

    if not math.isfinite(result):
        return None

    return float(result)
