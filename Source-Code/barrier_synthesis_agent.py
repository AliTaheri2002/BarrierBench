import re
import logging
import time
from utils import generate_samples_for_barrier_validation, validate_barrier_on_samples, parse_controller_expressions, substitute_controller_into_dynamics_for_samples
from barrier_verifier_agent import validate_barrier_with_agentic_smt
from barrier_retrieval_agent import BarrierRetrievalAgent
import anthropic

logger = logging.getLogger(__name__)


class BarrierSynthesisAgent:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514",
                 max_iterations: int = 5, dataset_json_path: str = "barrier_dataset.json"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.iteration_history = []

        self.retrieval_agent = BarrierRetrievalAgent(json_file_path=dataset_json_path)
        self.retrieval_agent.client = self.client
        self.retrieval_agent.model = self.model

    def synthesize_barrier_certificate(self, problem):
        start_time = time.time()
        self.iteration_history = []

        has_controller = bool(problem.get('controller_parameters', '').strip())

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n[Iteration {iteration}/{self.max_iterations}]")

            if has_controller:
                result = self.get_template_and_barrier_from_llm(problem, iteration, has_controller)
                if not result or len(result) != 2:
                    logger.warning(f"Failed to get barrier+controller in iteration {iteration}")
                    continue
                barrier_expr, controller_expr = result
                print(f"  Barrier:    {barrier_expr}")
                print(f"  Controller: {controller_expr}")
            else:
                barrier_expr = self.get_template_and_barrier_from_llm(problem, iteration, has_controller)
                controller_expr = None
                if not barrier_expr:
                    logger.warning(f"Failed to get barrier in iteration {iteration}")
                    continue
                print(f"  Barrier: {barrier_expr}")

            verification_result = self.verify_barrier(barrier_expr, problem, controller_expr if has_controller else None)

            if not verification_result:
                logger.warning(f"Verification failed in iteration {iteration}")
                self._add_to_history(iteration, barrier_expr, 0, "verification_error", controller_expr if has_controller else None)
                continue

            score = sum([verification_result['condition_1'], verification_result['condition_2'], verification_result['condition_3']])
            print(f"  Score: {score}/3")

            original_barrier = barrier_expr
            original_controller = controller_expr if has_controller else None
            original_verification = verification_result
            best_barrier = barrier_expr
            best_controller = controller_expr if has_controller else None
            best_score = score
            best_verification = verification_result
            iteration_refinements = []

            if score < 3:
                for refinement in range(1, 5):

                    if has_controller:
                        refined = self.refine_barrier(original_barrier, best_barrier, problem, original_verification,
                                                      refinement, iteration_refinements, has_controller, original_controller, best_controller)

                        if not refined or len(refined) != 2:
                            continue
                        refined_barrier, refined_controller = refined
                    else:
                        refined_barrier = self.refine_barrier(original_barrier, best_barrier, problem,
                                               original_verification, refinement, iteration_refinements, has_controller)
                        refined_controller = None
                        if not refined_barrier:
                            continue

                    ref_verification = self.verify_barrier(refined_barrier, problem,refined_controller if has_controller else None)
                    if not ref_verification:
                        continue

                    ref_score = sum([ref_verification['condition_1'], ref_verification['condition_2'], ref_verification['condition_3']])
                    print(f"  [Refinement {refinement}/4] Barrier: {refined_barrier}  Score: {ref_score}/3")

                    iteration_refinements.append({'refinement_num': refinement,'barrier': refined_barrier,
                                                'controller': refined_controller if has_controller else None,
                                                'base_barrier': best_barrier, 'base_controller': best_controller if has_controller else None,
                                                'verification': ref_verification, 'failed_conditions': self._get_failed_conditions(ref_verification, detailed=True)})

                    if ref_score > best_score:
                        best_barrier = refined_barrier
                        best_controller = refined_controller if has_controller else None
                        best_score = ref_score
                        best_verification = ref_verification

                    if ref_score == 3:
                        break

            self._add_to_history(iteration, best_barrier, best_score, best_verification, best_controller if has_controller else None)

            if best_score == 3:
                elapsed = time.time() - start_time
                print(f"  -> Valid barrier found!")

                self.retrieval_agent.store(problem=problem, barrier_certificate=best_barrier,
                                           template_type="llm_generated_with_controller" if has_controller else "llm_generated",
                                           controller_certificate=best_controller if has_controller else None)

                result = {'success': True, 'barrier_certificate': best_barrier,
                          'iteration_found': iteration, 'total_time': elapsed,
                          'verification_details': best_verification}

                if has_controller:
                    result['controller_certificate'] = best_controller
                return result

        elapsed = time.time() - start_time
        best_attempt = max(self.iteration_history, key=lambda x: x['score']) if self.iteration_history else None

        result = {'success': False, 'best_score': best_attempt['score'] if best_attempt else 0,
                  'best_barrier': best_attempt['barrier'] if best_attempt else None,
                  'total_time': elapsed, 'total_iterations': len(self.iteration_history)}

        if has_controller and best_attempt:
            result['best_controller'] = best_attempt.get('controller')

        return result

    def get_template_and_barrier_from_llm(self, problem, iteration, has_controller=False):
        dynamics = problem.get('dynamics', '')
        is_discrete = '[k+1]' in dynamics or '[k]' in dynamics
        system_type = "DISCRETE-TIME" if is_discrete else "CONTINUOUS-TIME"
        condition_3_desc = "B(f(x)) - B(x) ≤ 0 for all x in the state space" if is_discrete else "∇B(x)·f(x) < 0 on boundary"

        if iteration == 1:
            similar_case = self.retrieval_agent.find_most_similar(problem)
            if similar_case:
                context = f"""Related problem found:
        EXAMPLE: {similar_case['problem']}
        B(x): {similar_case['barrier']}

        WARNING: This is just an example - your solution may be completely different NOT ONLY in terms of coefficients, BUT ALSO in format and structure. you may make mistakes, so do not consider this as a solution. Please don't copy it

        """
            else:
                context = "No similar problems found. Analyze this problem fresh.\n\n"
        else:
            context = self._prepare_iteration_context(iteration, has_controller)

        if has_controller:
            controller_explanation_1 = f"""
        CONTROLLER SYNTHESIS: This system has control inputs {problem.get('controller_parameters')}. You need to design BOTH:
        1. Barrier certificate B(x) 
        2. Controller expressions for {problem.get('controller_parameters')}

        The controller u(x) will be substituted into dynamics to create closed-loop system.
        """
            controller_explanation_2 = """
        Design both barrier certificate B(x) AND controller expressions that work together to satisfy all conditions.

        CRITICAL: 
        - Use ONLY real numbers in both barrier and controller expressions. No variables like 'c' or 'ε'. 
        - Solve specifically for THIS problem with appropriate coefficients.
        - Controller must be implementable with realistic actuators
        - Ensure controller bounds are reasonable (avoid extremely large values)
        """
            barrier_controller_format = """
        BARRIER: [barrier expression with numbers only]
        CONTROLLER: [controller expressions for each parameter, comma-separated]"""
        else:
            controller_explanation_1 = ""
            controller_explanation_2 = """
        CRITICAL: Use ONLY real numbers in the barrier expression. No variables like 'c' or 'ε'. 
        Solve specifically for THIS problem with appropriate coefficients.
        """
            barrier_controller_format = """
        BARRIER: [expression with numbers only]"""

        prompt = f"""{context}Main Problem:
        - Dynamics: {problem.get('dynamics')}
        - Initial set: {problem.get('initial_set')}
        - Unsafe set: {problem.get('unsafe_set')}
        {controller_explanation_1}
        Design a barrier certificate B(x) that satisfies:
        1. B(x) ≤ 0 in initial set
        2. B(x) > 0 in unsafe set  
        3. {condition_3_desc}

        {"Be very careful - don't make it more complex than needed." if iteration == 1 else "Learn from previous failures. You can change structure of TEMPLATE if needed. In this step, the goal is to improve the structure of the templates, not refine the parameters." if has_controller else "Design a barrier certificate that satisfies all 3 conditions. Learn from previous failures. You can change structure of TEMPLATE if needed. In this step, the goal is to improve the structure of the templates, not refine the parameters."}

        {controller_explanation_2}

        Analyze carefully but be concise. Give precise answer without long explanations.

        Format your response as (don't make it bold):
        {barrier_controller_format}"""

        response = self.client.messages.create(model=self.model, max_tokens=1500 if has_controller else 1200, messages=[{"role": "user", "content": prompt}])
        content = response.content[0].text

        if has_controller:
            barrier_expr, controller_expr = self._extract(content, with_controller=True)
            if barrier_expr and controller_expr:
                return (barrier_expr, controller_expr)
            logger.warning("Failed to extract barrier and/or controller")
            return None
        else:
            barrier_expr = self._extract(content)
            if barrier_expr:
                return barrier_expr
            logger.warning("Failed to extract barrier")
            return None

    def refine_barrier(self, original_barrier, current_barrier, problem, verification, refinement_num,
                       iteration_refinements=None, has_controller=False, original_controller=None, current_controller=None):

        dynamics = problem.get('dynamics', '')
        is_discrete = '[k+1]' in dynamics or '[k]' in dynamics
        condition_3_desc = "B(f(x)) - B(x) ≤ 0 for all x" if is_discrete else "∇B(x)·f(x) < 0 on boundary"

        if refinement_num == 1:
            if has_controller:
                original_failed = self._get_failed_conditions(verification, detailed=True)
                refinement_context = f"Original barrier: {original_barrier}, Original controller: {original_controller} (failed: {original_failed})\n"
            else:
                original_failed = self._get_failed_conditions(verification, detailed=True)
                refinement_context = f"Original barrier: {original_barrier} (failed: {original_failed})\n"
        else:
            if has_controller:
                original_failed = self._get_failed_conditions(verification, detailed=True)
                refinement_context = f"Original barrier: {original_barrier}, Original controller: {original_controller} (failed: {original_failed})\n"
                if iteration_refinements:
                    for i, ref_data in enumerate(iteration_refinements, 1):
                        refinement_context += f"Refinement {i}: Barrier: {ref_data['barrier']}, Controller: {ref_data.get('controller', 'None')} (failed: {ref_data['failed_conditions']})\n"
            else:
                original_failed = self._get_failed_conditions(verification, detailed=True)
                refinement_context = f"Original barrier: {original_barrier} (failed: {original_failed})\n"
                if iteration_refinements:
                    for i, ref_data in enumerate(iteration_refinements, 1):
                        refinement_context += f"Refinement {i}: {ref_data['barrier']} (failed: {ref_data['failed_conditions']})\n"

        if has_controller:
            instruction = (
                "Try a different coefficient distribution for both barrier and controller. You can redistribute the coefficients between ALL terms, but DO NOT change structure"
                if refinement_num <= 2 else
                "Previous coefficient adjustments failed. Consider changing the barrier and/or controller structure if needed while keeping the same polynomial degree. You can modify the terms or their combinations."
            )
            controller_requirements = f"""
    CONTROLLER SYNTHESIS CONSTRAINTS:
    1. Controller parameters: {problem.get('controller_parameters')}
    2. Use smooth, bounded functions (avoid extremely large values)
    3. Controller must work harmoniously with the barrier
    4. Ensure closed-loop stability
    """
            response_format = """
    REFINED_BARRIER: [barrier expression with numbers only]
    REFINED_CONTROLLER: [controller expressions for each parameter, comma-separated]"""
        else:
            instruction = (
                "Try a different coefficient distribution. You can redistribute the coefficients between ALL terms (sometimes it is necessary for all terms to have different coefficients), but DO NOT change structure"
                if refinement_num <= 2 else
                "Previous coefficient adjustments failed. Consider changing the barrier structure if needed while keeping the same polynomial degree. You can modify the terms or their combinations."
            )
            controller_requirements = ""
            response_format = """
    REFINED_BARRIER: [expression with numbers only]"""

        prompt = f"""{refinement_context}

    Problem:
    - Dynamics: {problem.get('dynamics')}
    - Initial set: {problem.get('initial_set')}
    - Unsafe set: {problem.get('unsafe_set')}

    {instruction}
    {controller_requirements}
    Requirements:
    1. B(x) ≤ 0 in initial set
    2. B(x) > 0 in unsafe set  
    3. {condition_3_desc}

    Analyze carefully but be concise. Give precise answer without long explanations.

    Format your response as (don't make it bold):
    {response_format}"""

        response = self.client.messages.create(model=self.model, max_tokens=1200 if has_controller else 900, messages=[{"role": "user", "content": prompt}])
        content = response.content[0].text

        if has_controller:
            refined_barrier, refined_controller = self._extract(content, refined=True, with_controller=True)
            if refined_barrier and refined_controller:
                return (refined_barrier, refined_controller)
            logger.warning(f"Failed to extract refined barrier/controller from refinement {refinement_num}")
            return None
        else:
            refined_expr = self._extract(content, refined=True)
            if refined_expr:
                return refined_expr
            logger.warning(f"Failed to extract refined barrier from refinement {refinement_num}")
            return None

    def verify_barrier(self, barrier_expr, problem, controller_expr=None):
        working_problem = problem.copy()

        if controller_expr:
            controller_dict = parse_controller_expressions(controller_expr, problem)
            if controller_dict:
                closed_loop = substitute_controller_into_dynamics_for_samples(problem['dynamics'], controller_dict)
                if not closed_loop:
                    logger.warning("Failed to substitute controller into dynamics")
                    return None
                working_problem['dynamics'] = closed_loop
            else:
                logger.warning("Failed to parse controller expressions")
                return None

        # Sample gatekeeper
        samples = generate_samples_for_barrier_validation(working_problem, num_samples=5000)
        sample_validation = validate_barrier_on_samples(barrier_expr, working_problem, samples)

        if not sample_validation['success']:
            logger.warning("Sample validation failed")
            return None

        sample_conditions = sample_validation['conditions_satisfied']
        if not all(sample_conditions):
            logger.warning("Gate-keeper rejection")
            sample_counts = {}
            if 'condition_details' in sample_validation:
                details = sample_validation['condition_details']
                sample_counts['condition_1'] = details.get('condition_1_failed_count', 0)
                sample_counts['condition_2'] = details.get('condition_2_failed_count', 0)
                sample_counts['condition_3'] = details.get('condition_3_failed_count', 0)
            return {'condition_1': sample_conditions[0], 'condition_2': sample_conditions[1],
                    'condition_3': sample_conditions[2], 'sample_counts': sample_counts}


        smt_validation = validate_barrier_with_agentic_smt(barrier_expr, working_problem['initial_set'],
                         working_problem['unsafe_set'], working_problem['dynamics'], self.client, self.model)

        if smt_validation['success']:
            return smt_validation['conditions']
        else:
            logger.warning(f"SMT failed: {smt_validation.get('error')}")
            return None

    def _prepare_iteration_context(self, iteration, has_controller=False):
        if not self.iteration_history:
            return ""

        if has_controller:
            context = "Previous barrier+controller attempts failed:\n"
            for hist in self.iteration_history[-3:]:
                failed_conditions = self._get_failed_conditions(hist.get('verification', {}), detailed=True)
                context += f"- Barrier: {hist['barrier']}, Controller: {hist.get('controller', 'None')} (failed: {failed_conditions})\n"
        else:
            context = "Previous attempts failed:\n"
            for hist in self.iteration_history[-3:]:
                failed_conditions = self._get_failed_conditions(hist.get('verification', {}), detailed=True)
                context += f"- Tried: {hist['barrier']} (failed: {failed_conditions})\n"

        context += f"\nImprove the {'barrier+controller' if has_controller else 'barrier'} structure to satisfy all conditions.\n\n"
        return context

    def _get_failed_conditions(self, verification, detailed=False):
        if not verification or not isinstance(verification, dict):
            return "unknown"

        failed = []
        has_sample_info = 'sample_counts' in verification

        if not verification.get('condition_1', True):
            if detailed and has_sample_info:
                count = verification.get('sample_counts', {}).get('condition_1', 0)
                failed.append(f"initial set ({count} samples)")
            else:
                failed.append("initial set")

        if not verification.get('condition_2', True):
            if detailed and has_sample_info:
                count = verification.get('sample_counts', {}).get('condition_2', 0)
                failed.append(f"unsafe set ({count} samples)")
            else:
                failed.append("unsafe set")

        if not verification.get('condition_3', True):
            if detailed and has_sample_info:
                count = verification.get('sample_counts', {}).get('condition_3', 0)
                failed.append(f"lie derivative ({count} samples)")
            else:
                failed.append("lie derivative")

        return ", ".join(failed) if failed else "none"

    def _add_to_history(self, iteration, barrier, score, verification=None, controller=None):
        entry = {'iteration': iteration, 'barrier': barrier, 'score': score, 'verification': verification}
        if controller is not None:
            entry['controller'] = controller
        self.iteration_history.append(entry)

    def _extract(self, content, refined=False, with_controller=False):
        barrier_patterns = []
        if refined or not with_controller:
            barrier_patterns.append(r'REFINED_BARRIER\s*:\s*(.+?)(?:\n|$)')
        barrier_patterns.append(r'BARRIER\s*:\s*(.+?)(?:\n|$)')
        if not refined:
            barrier_patterns += [
                r'BARRIER_CERTIFICATE\s*:\s*(.+?)(?:\n|$)',
                r'B\(x\)\s*=\s*(.+?)(?:\n|$)'
            ]

        barrier_expr = None
        for pattern in barrier_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                cleaned = self._clean_expression(matches[0].strip())
                if cleaned and self._validate_expression(cleaned):
                    barrier_expr = cleaned
                    break

        # fallback line scan — only for barrier-only extraction
        if barrier_expr is None and not with_controller:
            for line in content.split('\n'):
                line = line.strip()
                if any(w in line.lower() for w in ['compute', 'derivative', 'verify', 'analysis', 'let me', 'boundary', 'violation', 'check']):
                    continue
                if (len(line) > 10 and re.search(r'x\d+', line) and any(op in line for op in ['+', '-', '*']) and
                        re.search(r'\d', line) and not any(w in line.lower() for w in ['barrier', 'certificate', 'template', 'function', 'expression'])):
                    cleaned = self._clean_expression(line)
                    if cleaned and self._validate_expression(cleaned):
                        barrier_expr = cleaned
                        break

        if not with_controller:
            return barrier_expr

        controller_patterns = []
        if refined:
            controller_patterns.append(r'REFINED_CONTROLLER\s*:\s*([\s\S]+?)(?:\n\n|\n[A-Z]|$)')
        controller_patterns += [
            r'CONTROLLER\s*:\s*([\s\S]+?)(?:\n\n|\n[A-Z]|$)',
        ]
        if not refined:
            controller_patterns.append(r'CONTROL\s*:\s*([\s\S]+?)(?:\n\n|\n[A-Z]|$)')

        controller_expr = None
        for pattern in controller_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                cleaned = self._clean_controller_expression(matches[0].strip())
                if cleaned:
                    controller_expr = cleaned
                    break

        return barrier_expr, controller_expr

    def _clean_expression(self, expr):
        if not expr:
            return ""

        expr = expr.replace('`', '').strip()
        expr = re.sub(r'^.*?BARRIER.*?:\s*', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'^.*?B\(.*?\)\s*=\s*', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'^.*?[:=]\s*', '', expr)

        for u, r in [('²','**2'),('³','**3'),('⁴','**4'),('⁰','**0'),('¹','**1'),('⁵','**5'),('⁶','**6'),('⁷','**7'),('⁸','**8'),('⁹','**9')]:
            expr = expr.replace(u, r)

        expr = re.sub(r'\^(\d+)', r'**\1', expr)
        expr = expr.replace('^', '**')
        expr = re.sub(r'(\w)(\()', r'\1*\2', expr)
        expr = re.sub(r'(\))(\w)', r'\1*\2', expr)
        expr = re.sub(r'\((-?\d+\.?\d*)\)', r'\1', expr)
        expr = re.sub(r'(\d)([x])', r'\1*\2', expr)
        expr = re.sub(r'(\d)\s+([x])', r'\1*\2', expr)
        expr = re.sub(r'\s+', ' ', expr).strip()
        expr = expr.rstrip('.,;:')
        expr = re.sub(r'\*\*+', '**', expr)
        expr = re.sub(r'\++', '+', expr)
        expr = re.sub(r'--+', '-', expr)
        expr = re.sub(r'[^\w\s\+\-\*\(\)\.\/]', '', expr)

        return expr

    def _clean_controller_expression(self, expr):
        if not expr:
            return ""

        expr = expr.replace('`', '').strip()
        expr = re.sub(r'\[k\+1\]', '', expr)
        expr = re.sub(r'\[k\]', '', expr)
        expr = re.sub(r'^.*?CONTROLLER.*?:\s*', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'^.*?CONTROL.*?:\s*', '', expr, flags=re.IGNORECASE)

        for u, r in [('²','**2'),('³','**3')]:
            expr = expr.replace(u, r)
        expr = re.sub(r'\^(\d+)', r'**\1', expr)
        expr = expr.replace('^', '**')
        expr = re.sub(r'(\d)([x])', r'\1*\2', expr)
        expr = re.sub(r'(\d)\s+([x])', r'\1*\2', expr)
        expr = re.sub(r'\s+', ' ', expr).strip()
        expr = expr.rstrip('.,;:')

        return expr

    def _validate_expression(self, expr):
        if not expr or len(expr) < 3:
            return False
        if not re.search(r'x\d+', expr):
            return False
        if not re.search(r'\d', expr):
            return False

        test_expr = expr
        for i in range(1, 10):
            test_expr = test_expr.replace(f'x{i}', '1')
        compile(test_expr, '<string>', 'eval')
        return True
