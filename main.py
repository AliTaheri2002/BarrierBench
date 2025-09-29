import os
import sys
import logging
import json
import time
import math
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from copy import deepcopy
import anthropic
import numpy as np

# Import existing modules
from barrier_verification import get_detailed_condition_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BarrierDataset:
    """Simple dataset for storing solved barrier certificate problems"""
    
    def __init__(self, json_file_path: str = "barrier_dataset.json"):
        self.json_file_path = json_file_path
        self.test_cases = []
        self.load_dataset_from_json()
        logger.info(f"Loaded {len(self.test_cases)} solved problems from dataset")

    def load_dataset_from_json(self):
        """Load existing problems from JSON"""
        try:
            if os.path.exists(self.json_file_path):
                with open(self.json_file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # File is not empty
                        data = json.loads(content)
                        self.test_cases = data.get('solved_problems', [])
                    else:  # Empty file
                        self.test_cases = []
                        self._save_dataset_to_json()
            else:
                self.test_cases = []
                self._save_dataset_to_json()
        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}")
            self.test_cases = []
            # Create fresh JSON file
            self._save_dataset_to_json()

    def save_successful_barrier_to_json(self, problem: Dict[str, Any], barrier_certificate: str, 
                                  template_type: str, controller_certificate: str = None):
        """Save successful solution"""
        try:
            new_record = {
                'problem': problem,
                'barrier': barrier_certificate
            }
            
            # Add controllers field if this is a controller problem
            if controller_certificate is not None:
                new_record['controllers'] = controller_certificate
            
            new_record['template_type'] = template_type

            self.test_cases.append(new_record)
            self._save_dataset_to_json()
            logger.info(f"Saved to dataset: {barrier_certificate}")
            if controller_certificate:
                logger.info(f"With controllers: {controller_certificate}")
        except Exception as e:
            logger.error(f"Failed to save: {e}")

    def find_most_similar(self, target_problem: Dict[str, Any], threshold: float = 0.7) -> Tuple[Optional[Dict], float]:
        """Critical filter approach: Hard filter on fundamental differences + LLM selection"""
        
        print(f"\nCritical Filter: Starting with {len(self.test_cases)} candidates")
        
        # Extract target's critical features
        target_features = self._extract_critical_features(target_problem)
        if not target_features:
            print("Failed to extract target features")
            return None, 0.0
        
        print(f"Target features: {target_features}")
        
        # Step 1: Apply hard filters
        compatible_candidates = []
        for i, case in enumerate(self.test_cases):
            candidate_features = self._extract_critical_features(case['problem'])
            
            if self._are_fundamentally_compatible(target_features, candidate_features):
                compatible_candidates.append(case)
            else:
                print(f"  Filtered out candidate {i+1}: incompatible features {candidate_features}")
        
        print(f"After filtering: {len(compatible_candidates)} candidates remain")
        
        if not compatible_candidates:
            print("No compatible candidates found")
            return None, 0.0
        
        if len(compatible_candidates) == 1:
            print("Single compatible candidate - using directly")
            return compatible_candidates[0], 0.95
        
        # Step 2: LLM selection on remaining candidates
        print(f"Sending {len(compatible_candidates)} candidates to LLM for selection")
        best_candidate = self._llm_select_from_compatible(target_problem, compatible_candidates)
        
        if best_candidate:
            return best_candidate, 0.85
        else:
            # Fallback to first compatible
            return compatible_candidates[0], 0.75

    def _extract_critical_features(self, problem: Dict[str, Any]) -> Optional[Dict]:
        """Extract the absolutely critical features that determine barrier design approach"""
        try:
            dynamics = problem.get('dynamics', '')
            initial_set = problem.get('initial_set', {})
            unsafe_set = problem.get('unsafe_set', {})
            
            features = {
                # 1. System Dimension (2D vs 3D vs 4D+ fundamentally different)
                'dimension': self._get_system_dimension(dynamics),
                
                # 2. Time Domain (continuous vs discrete need different approaches)  
                'time_domain': self._get_time_domain(dynamics),
                
                # 3. Linearity (linear vs nonlinear barriers are completely different)
                'linearity': self._get_linearity(dynamics),
                
                # 4. Set Topology (ball vs box vs complement need different barrier approaches)
                'set_topology': self._get_set_topology(initial_set, unsafe_set)
            }
            
            return features
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return None

    def _get_system_dimension(self, dynamics: str) -> int:
        """System dimension - 2D vs 3D vs 4D+ is fundamental"""
        variables = set(re.findall(r'x(\d+)', dynamics))
        return len(variables)

    def _get_time_domain(self, dynamics: str) -> str:
        """Continuous time vs discrete time"""
        if 'dt' in dynamics or 'd/dt' in dynamics:
            return 'continuous'
        elif '[k+1]' in dynamics or '[k]' in dynamics:
            return 'discrete'
        else:
            return 'unknown'

    def _get_linearity(self, dynamics: str) -> str:
        """Linear vs nonlinear systems need fundamentally different barriers"""
        if re.search(r'x\d+\*\*[2-9]', dynamics) or re.search(r'x\d+\*x\d+', dynamics):
            return 'nonlinear'
        elif re.search(r'x\d+', dynamics):
            return 'linear'
        else:
            return 'unknown'

    def _get_set_topology(self, initial_set: Dict, unsafe_set: Dict) -> str:
        """Set topology affects barrier shape requirements"""
        init_type = initial_set.get('type', 'unknown')
        unsafe_type = unsafe_set.get('type', 'unknown')
        unsafe_complement = unsafe_set.get('complement', False)
        
        # Create a topology signature
        topology = f"{init_type}_to_{'complement_' if unsafe_complement else ''}{unsafe_type}"
        return topology

    def _are_fundamentally_compatible(self, features1: Dict, features2: Dict) -> bool:
        """Check if two problems are fundamentally compatible for barrier transfer"""
        
        # Critical compatibility checks - if any fail, problems are incompatible
        critical_checks = [
            # 1. Same dimension (2D vs 3D vs 4D+ barrier design completely different)
            features1.get('dimension') == features2.get('dimension'),
            
            # 2. Same time domain (continuous vs discrete need different approaches)
            features1.get('time_domain') == features2.get('time_domain'),
            
            # 3. Same linearity (linear vs nonlinear barriers fundamentally different)
            features1.get('linearity') == features2.get('linearity'),
            
            # 4. Compatible topology (some set combinations are impossible)
            self._topology_compatible(features1.get('set_topology'), features2.get('set_topology'))
        ]
        
        # ALL critical features must be compatible
        is_compatible = all(critical_checks)
        
        if not is_compatible:
            print(f"    Incompatible: dim:{critical_checks[0]} time:{critical_checks[1]} "
                  f"linear:{critical_checks[2]} topology:{critical_checks[3]}")
        
        return is_compatible

    def _topology_compatible(self, topo1: str, topo2: str) -> bool:
        """Check if set topologies are compatible"""
        # Exact match is always compatible
        if topo1 == topo2:
            return True
        
        # Some topology transfers might work
        # E.g., ball_to_ball might work with ball_to_complement_ball
        base1 = topo1.replace('complement_', '')
        base2 = topo2.replace('complement_', '')
        
        # Same base topology might be transferable
        return base1 == base2

    def _llm_select_from_compatible(self, target_problem: Dict[str, Any], 
                                   compatible_candidates: List[Dict]) -> Optional[Dict]:
        """Use LLM to select the best from fundamentally compatible candidates"""
        
        # Prepare compact representation of each candidate
        candidates_text = ""
        for i, candidate in enumerate(compatible_candidates, 1):
            prob = candidate['problem']
            candidates_text += f"\nCANDIDATE {i}:\n"
            candidates_text += f"Dynamics: {prob.get('dynamics')}\n"  
            candidates_text += f"Initial: {prob.get('initial_set')}\n"
            candidates_text += f"Unsafe: {prob.get('unsafe_set')}\n"
            candidates_text += f"Successful barrier: {candidate['barrier']}\n"
        
        prompt = f"""TARGET PROBLEM:
Dynamics: {target_problem.get('dynamics')}
Initial set: {target_problem.get('initial_set')}
Unsafe set: {target_problem.get('unsafe_set')}

COMPATIBLE CANDIDATES (all are fundamentally similar):{candidates_text}

Which candidate has the most similar problem type and structure to the target problem?
Focus on: system structure, problem type, and mathematical pattern similarity.

Answer with only the candidate number (1, 2, 3, etc.): """

        try:
            # In practice, you would call your LLM client here
            # For now, return the first candidate as fallback
            print(f"LLM prompt would be {len(prompt)} characters")
            print("LLM selection placeholder - returning first compatible candidate")
            return compatible_candidates[0]
            
        except Exception as e:
            print(f"LLM selection failed: {e}")
            return compatible_candidates[0]  # Fallback

    def _save_dataset_to_json(self):
        """Save dataset to JSON"""
        try:
            with open(self.json_file_path, 'w', encoding='utf-8') as f:
                json.dump({'solved_problems': self.test_cases}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")


class SimplifiedBarrierSynthesis:
    """
    Simplified LLM-driven Barrier Certificate Synthesis
    
    - LLM designs templates
    - 5 iterations max
    - 4 refinements per iteration
    - Learning from previous failures
    """
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", 
                 max_iterations: int = 5, dataset_json_path: str = "barrier_dataset.json"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.dataset = BarrierDataset(json_file_path=dataset_json_path)
        self.iteration_history = []

    def synthesize_barrier_certificate(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Main synthesis pipeline with controller support"""
        logger.info("Starting Simplified Barrier Synthesis")
        start_time = time.time()
        self.iteration_history = []
        
        # Controller flag detection
        has_controller = len(problem.get('controller_parameters', '').strip().split(',')) > 0 and problem.get('controller_parameters', '').strip()
        if has_controller:
            logger.info(f"Controller synthesis enabled for parameters: {problem.get('controller_parameters')}")
        else:
            logger.info("Standard barrier synthesis (no controller)")
        
        try:
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"\n=== ITERATION {iteration}/{self.max_iterations} ===")
                
                # Step 1: Get template from LLM (now handles controller too)
                if has_controller:
                    template_and_expressions = self._get_template_and_barrier_from_llm(problem, iteration, has_controller)
                    if not template_and_expressions or len(template_and_expressions) != 2:
                        logger.warning(f"Failed to get barrier+controller in iteration {iteration}")
                        continue
                    barrier_expr, controller_expr = template_and_expressions
                    logger.info(f"LLM suggested barrier: {barrier_expr}")
                    logger.info(f"LLM suggested controller: {controller_expr}")
                else:
                    barrier_expr = self._get_template_and_barrier_from_llm(problem, iteration, has_controller)
                    controller_expr = None
                    if not barrier_expr:
                        logger.warning(f"Failed to get barrier in iteration {iteration}")
                        continue
                    logger.info(f"LLM suggested: {barrier_expr}")
                
                # Step 2: Verify (controller-aware if needed)
                verification_result = self._verify_barrier(barrier_expr, problem, controller_expr if has_controller else None)
                
                if not verification_result:
                    logger.warning(f"Verification failed in iteration {iteration}")
                    self._add_to_history(iteration, barrier_expr, 0, "verification_error", controller_expr if has_controller else None)
                    continue
                
                score = sum([
                    verification_result['condition_1'],
                    verification_result['condition_2'], 
                    verification_result['condition_3']
                ])
                
                logger.info(f"Initial score: {score}/3")
                
                # Step 3: Refinements (up to 4) - controller-aware
                original_barrier = barrier_expr
                original_controller = controller_expr if has_controller else None
                best_barrier = barrier_expr
                best_controller = controller_expr if has_controller else None
                best_score = score
                best_verification = verification_result
                
                # Track refinements for this iteration
                iteration_refinements = []
                
                if score < 3:
                    for refinement in range(1, 5):  # 4 refinements max
                        logger.info(f"Refinement {refinement}/4")
                        
                        # Prepare refinement context (controller-aware)
                        if has_controller:
                            refined_expressions = self._refine_barrier(
                                original_barrier, best_barrier, problem, best_verification, 
                                refinement, iteration_refinements, has_controller, 
                                original_controller, best_controller
                            )
                            if not refined_expressions or len(refined_expressions) != 2:
                                continue
                            refined_barrier, refined_controller = refined_expressions
                        else:
                            refined_barrier = self._refine_barrier(
                                original_barrier, best_barrier, problem, best_verification, 
                                refinement, iteration_refinements, has_controller
                            )
                            refined_controller = None
                            if not refined_barrier:
                                continue
                        
                        ref_verification = self._verify_barrier(
                            refined_barrier, problem, 
                            refined_controller if has_controller else None
                        )
                        if not ref_verification:
                            continue
                        
                        # Store this refinement with its verification details
                        iteration_refinements.append({
                            'refinement_num': refinement,
                            'barrier': refined_barrier,
                            'controller': refined_controller if has_controller else None,
                            'base_barrier': best_barrier,
                            'base_controller': best_controller if has_controller else None,
                            'verification': ref_verification,
                            'failed_conditions': self._get_failed_conditions(ref_verification)
                        })
                        
                        ref_score = sum([
                            ref_verification['condition_1'],
                            ref_verification['condition_2'],
                            ref_verification['condition_3']
                        ])
                        
                        logger.info(f"Refinement {refinement} score: {ref_score}/3")
                        
                        if ref_score >= best_score:
                            best_barrier = refined_barrier
                            best_controller = refined_controller if has_controller else None
                            best_score = ref_score
                            best_verification = ref_verification
                        
                        if ref_score == 3:
                            break
                
                # Store result
                self._add_to_history(iteration, best_barrier, best_score, best_verification, best_controller if has_controller else None)
                
                # Success check
                if best_score == 3:
                    elapsed_time = time.time() - start_time
                    logger.info(f"SUCCESS! Found valid {'barrier+controller' if has_controller else 'barrier'} in iteration {iteration}")
                    
                    # Save to dataset
                    try:
                        self.dataset.save_successful_barrier_to_json(
                            problem=problem,
                            barrier_certificate=best_barrier,
                            template_type="llm_generated_with_controller" if has_controller else "llm_generated",
                            controller_certificate=best_controller if has_controller else None
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save to dataset: {e}")
                    
                    result = {
                        'success': True,
                        'barrier_certificate': best_barrier,
                        'iteration_found': iteration,
                        'total_time': elapsed_time,
                        'verification_details': best_verification
                    }
                    
                    if has_controller:
                        result['controller_certificate'] = best_controller
                    
                    return result
            
            # No success
            elapsed_time = time.time() - start_time
            best_attempt = max(self.iteration_history, key=lambda x: x['score']) if self.iteration_history else None
            
            result = {
                'success': False,
                'best_score': best_attempt['score'] if best_attempt else 0,
                'best_barrier': best_attempt['barrier'] if best_attempt else None,
                'total_time': elapsed_time,
                'total_iterations': len(self.iteration_history)
            }
            
            if has_controller and best_attempt:
                result['best_controller'] = best_attempt.get('controller')
            
            return result
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time
            }
    
    def _get_template_and_barrier_from_llm(self, problem: Dict[str, Any], iteration: int, has_controller: bool = False):
        """Get template and barrier from LLM with optional controller support"""
        
        # Check for similar problems
        if iteration == 1:
            similar_case, similarity = self.dataset.find_most_similar(problem)
            context = ""
            if similar_case and similarity > 0.99:
                context = f"""Related problem found:
    EXAMPLE: {similar_case['problem']}
    B(x): {similar_case['barrier']}

    WARNING: This is just an example - your solution may be completely different NOT ONLY in terms of coefficients, BUT ALSO in format and structure. you may make mistakes, so do not consider this as a solution. Please don't copy it

    """
            else:
                context = "No similar problems found. Analyze this problem fresh.\n\n"
        else:
            # Provide previous failures
            context = self._prepare_iteration_context(iteration, has_controller)
        
        # Controller-aware prompts
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
    3. ∇B·f < 0 on boundary

    {"Design a barrier certificate and controller that satisfy all 3 conditions. Be very careful - don't make it more complex than needed." if iteration == 1 else "Design barrier certificate and controller that satisfy all 3 conditions. Learn from previous failures. You can change structure of TEMPLATE if needed. In this step, the goal is to improve the structure of the templates, not refine the parameters." if has_controller else "Design a barrier certificate that satisfies all 3 conditions. Learn from previous failures. You can change structure of TEMPLATE if needed. In this step, the goal is to improve the structure of the templates, not refine the parameters."}

    {controller_explanation_2}

    Analyze carefully but be concise. Give precise answer without long explanations.

    {barrier_controller_format}"""

        print("\n" + "="*80)
        print(f"TEMPLATE/BARRIER{'_CONTROLLER' if has_controller else ''} PROMPT (Iteration {iteration}):")
        print("="*80)
        print(prompt)
        print("="*80)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500 if has_controller else 1200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            print("\n" + "="*80)
            print(f"CLAUDE TEMPLATE/BARRIER{'_CONTROLLER' if has_controller else ''} RESPONSE:")
            print("="*80)
            print(content)
            print("="*80)
            
            if has_controller:
                barrier_expr, controller_expr = self._extract_barrier_and_controller_from_response(content)
                
                if barrier_expr and controller_expr:
                    logger.info(f"Successfully extracted barrier: {barrier_expr}")
                    logger.info(f"Successfully extracted controller: {controller_expr}")
                    return (barrier_expr, controller_expr)
                else:
                    logger.warning("Failed to extract barrier and/or controller from response")
                    return None
            else:
                barrier_expr = self._extract_barrier_from_response(content)
                
                if barrier_expr:
                    logger.info(f"Successfully extracted barrier: {barrier_expr}")
                    return barrier_expr
                else:
                    logger.warning("Failed to extract barrier from response")
                    return None
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return None
    
    def _prepare_iteration_context(self, iteration: int, has_controller: bool = False) -> str:
        """Prepare context from previous failures with controller support"""
        if not self.iteration_history:
            return ""
        
        # Get FINAL best barriers/controllers from each previous iteration
        if has_controller:
            context = "Previous barrier+controller attempts failed:\n"
            for hist in self.iteration_history[-3:]:  # Last 3 iterations max
                failed_conditions = self._get_failed_conditions(hist.get('verification', {}))
                barrier = hist['barrier']
                controller = hist.get('controller', 'None')
                context += f"- Barrier: {barrier}, Controller: {controller} (failed: {failed_conditions})\n"
        else:
            context = "Previous attempts failed:\n"
            for hist in self.iteration_history[-3:]:  # Last 3 iterations max
                failed_conditions = self._get_failed_conditions(hist.get('verification', {}))
                context += f"- Tried: {hist['barrier']} (failed: {failed_conditions})\n"
        
        # Add general guidance
        context += f"\nImprove the {'barrier+controller' if has_controller else 'barrier'} structure to satisfy all conditions.\n\n"
        return context

    def _get_failed_conditions(self, verification: Dict) -> str:
        """Get description of failed conditions"""
        if not verification or not isinstance(verification, dict):
            return "unknown"
        
        failed = []
        if not verification.get('condition_1', True):
            failed.append("initial set")
        if not verification.get('condition_2', True):
            failed.append("unsafe set")
        if not verification.get('condition_3', True):
            failed.append("lie derivative")
        
        return ", ".join(failed) if failed else "none"

    def _refine_barrier(self, original_barrier: str, current_barrier: str, problem: Dict[str, Any], 
                   verification: Dict[str, Any], refinement_num: int, 
                   iteration_refinements: List[Dict] = None, has_controller: bool = False,
                   original_controller: str = None, current_controller: str = None):
        """Refine barrier with optional controller based on failures"""
        
        failed_conditions = self._get_failed_conditions(verification)
        
        # Prepare refinement context (controller-aware)
        if refinement_num == 1:
            if has_controller:
                original_failed = self._get_failed_conditions(verification)
                refinement_context = f"Original barrier: {original_barrier}, Original controller: {original_controller} (failed: {original_failed})\n"
            else:
                original_failed = self._get_failed_conditions(verification)
                refinement_context = f"Original barrier: {original_barrier} (failed: {original_failed})\n"
        else:
            if has_controller:
                original_failed = self._get_failed_conditions(verification)
                refinement_context = f"Original barrier: {original_barrier}, Original controller: {original_controller} (failed: {original_failed})\n"
                
                if iteration_refinements:
                    for i, ref_data in enumerate(iteration_refinements, 1):
                        ref_barrier = ref_data['barrier']
                        ref_controller = ref_data.get('controller', 'None')
                        ref_failed = ref_data['failed_conditions']
                        refinement_context += f"Refinement {i}: Barrier: {ref_barrier}, Controller: {ref_controller} (failed: {ref_failed})\n"
            else:
                original_failed = self._get_failed_conditions(verification)
                refinement_context = f"Original barrier: {original_barrier} (failed: {original_failed})\n"
                
                if iteration_refinements:
                    for i, ref_data in enumerate(iteration_refinements, 1):
                        ref_barrier = ref_data['barrier']
                        ref_failed = ref_data['failed_conditions']
                        refinement_context += f"Refinement {i}: {ref_barrier} (failed: {ref_failed})\n"
        
        # Different instructions based on refinement number and controller presence
        if has_controller:
            if refinement_num <= 2:
                instruction = "Try a different coefficient distribution for both barrier and controller. You can redistribute the coefficients between ALL terms, but DO NOT change structure"
            else:
                instruction = "Previous coefficient adjustments failed. Consider changing the barrier and/or controller structure if needed while keeping the same polynomial degree. You can modify the terms or their combinations."
            
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
            if refinement_num <= 2:
                instruction = "Try a different coefficient distribution. You can redistribute the coefficients between ALL terms (sometimes it is necessary for all terms to have different coefficients), but DO NOT change structure"
            else:
                instruction = "Previous coefficient adjustments failed. Consider changing the barrier structure if needed while keeping the same polynomial degree. You can modify the terms or their combinations."
            
            controller_requirements = ""
            response_format = """
    REFINED_BARRIER: [expression with numbers only]"""

        prompt = f"""{refinement_context}
    This {'barrier+controller combination' if has_controller else 'barrier'} violates: {failed_conditions}

    Problem:
    - Dynamics: {problem.get('dynamics')}
    - Initial set: {problem.get('initial_set')}
    - Unsafe set: {problem.get('unsafe_set')}

    {instruction}
    {controller_requirements}
    Requirements:
    1. B(x) ≤ 0 in initial set
    2. B(x) > 0 in unsafe set  
    3. ∇B(x)·f(x) < 0 on boundary

    Analyze carefully but be concise. Give precise answer without long explanations.

    {response_format}"""

        print("\n" + "="*80)
        print(f"REFINEMENT{'_CONTROLLER' if has_controller else ''} PROMPT (Attempt {refinement_num}):")
        print("="*80)
        print(prompt)
        print("="*80)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1200 if has_controller else 900,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            print("\n" + "="*80)
            print(f"CLAUDE REFINEMENT{'_CONTROLLER' if has_controller else ''} RESPONSE (Attempt {refinement_num}):")
            print("="*80)
            print(content)
            print("="*80)
            
            if has_controller:
                refined_barrier, refined_controller = self._extract_refined_barrier_and_controller_from_response(content)
                
                if refined_barrier and refined_controller:
                    logger.info(f"Refinement {refinement_num} produced barrier: {refined_barrier}")
                    logger.info(f"Refinement {refinement_num} produced controller: {refined_controller}")
                    return (refined_barrier, refined_controller)
                else:
                    logger.warning(f"Failed to extract refined barrier and/or controller from refinement {refinement_num}")
                    return None
            else:
                refined_expr = self._extract_barrier_from_response(content)
                
                if refined_expr:
                    logger.info(f"Refinement {refinement_num} produced: {refined_expr}")
                    return refined_expr
                else:
                    logger.warning(f"Failed to extract refined barrier from refinement {refinement_num}")
                    return None
            
        except Exception as e:
            logger.error(f"Refinement error: {e}")
            return None
    
    
    def _verify_barrier(self, barrier_expr: str, problem: Dict[str, Any], controller_expr: str = None) -> Optional[Dict[str, Any]]:
        """Verify barrier with sample gatekeeper + SMT, with optional controller support"""
        try:
            # Check dimensions
            system_vars = self._extract_system_variables(problem.get('dynamics', ''))
            barrier_vars = self._extract_barrier_variables(barrier_expr)
            
            missing_vars = set(system_vars) - set(barrier_vars)
            if missing_vars:
                logger.error(f"Missing variables: {missing_vars}")
                return None

            # Create working problem with closed-loop dynamics BEFORE sample generation
            working_problem = problem.copy()
            if controller_expr:
                controller_dict = self._parse_controller_expressions(controller_expr, problem)
                if controller_dict:
                    original_dynamics = problem['dynamics']
                    closed_loop_dynamics = self._substitute_controller_into_dynamics(original_dynamics, controller_dict)
                    working_problem['dynamics'] = closed_loop_dynamics
                    logger.info(f"Using closed-loop dynamics for samples: {closed_loop_dynamics}")
                else:
                    logger.warning("Failed to parse controller expressions")
                    return None

            # Sample gatekeeper - now uses closed-loop dynamics if controller present
            try:
                from data_structures import generate_samples_for_barrier_validation, validate_barrier_on_samples
                
                # Generate samples with the working problem (closed-loop if controller present)
                samples = generate_samples_for_barrier_validation(working_problem, num_samples=5000)
                
                # Validate (controller_expr=None since dynamics already closed-loop)
                sample_validation = validate_barrier_on_samples(barrier_expr, working_problem, samples)
                
                if not sample_validation['success']:
                    logger.warning("Sample validation failed")
                    return None
                
                sample_conditions = sample_validation['conditions_satisfied']
                if not all(sample_conditions):
                    logger.warning("Gate-keeper rejection")
                    return {
                        'condition_1': sample_conditions[0],
                        'condition_2': sample_conditions[1],
                        'condition_3': sample_conditions[2]
                    }
                
                logger.info("Gate-keeper approved, proceeding to SMT")
                
            except Exception as e:
                logger.warning(f"Sample validation unavailable: {e}")

            # SMT verification (use working_problem which has closed-loop dynamics)
            smt_validation = get_detailed_condition_results(
                barrier_expr,
                working_problem['initial_set'],
                working_problem['unsafe_set'], 
                working_problem['dynamics']
            )
            
            if smt_validation['success']:
                return smt_validation['conditions']
            else:
                logger.warning(f"SMT failed: {smt_validation.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return None
    
    def _extract_system_variables(self, dynamics: str) -> List[str]:
        """Extract variables from dynamics"""
        if not dynamics:
            return []
        variables = set(re.findall(r'\bx\d+\b', dynamics))
        return sorted(list(variables), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)

    def _extract_barrier_variables(self, barrier_expr: str) -> List[str]:
        """Extract variables from barrier"""
        if not barrier_expr:
            return []
        variables = set(re.findall(r'\bx\d+\b', barrier_expr))
        return sorted(list(variables), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)

    def _extract_barrier_from_response(self, content: str) -> Optional[str]:
        """Extract barrier from LLM response"""
        # Try structured patterns first - in order of priority
        patterns = [
            r'REFINED_BARRIER\s*:\s*(.+?)(?:\n|$)',  # Highest priority for refinements
            r'BARRIER\s*:\s*(.+?)(?:\n|$)',
            r'BARRIER_CERTIFICATE\s*:\s*(.+?)(?:\n|$)',
            r'B\(x\)\s*=\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                expr = matches[0].strip()
                cleaned = self._clean_expression(expr)
                if cleaned and self._validate_expression(cleaned):
                    return cleaned
        
        # Fallback: look for mathematical expressions, but exclude analysis lines
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            # Skip lines that are clearly analysis/explanation
            if any(word in line.lower() for word in ['compute', 'derivative', 'verify', 'analysis', 'let me', 'boundary', 'violation', 'check']):
                continue
                
            if (len(line) > 10 and 
                re.search(r'x\d+', line) and 
                any(op in line for op in ['+', '-', '*']) and
                re.search(r'\d', line) and
                not any(word in line.lower() for word in ['barrier', 'certificate', 'template', 'function', 'expression'])):
                
                cleaned = self._clean_expression(line)
                if cleaned and self._validate_expression(cleaned):
                    return cleaned
        
        return None

    def _clean_expression(self, expr: str) -> str:
        """Clean expression more thoroughly"""
        if not expr:
            return ""
        
        # Basic cleaning
        expr = expr.replace('`', '').strip()
        
        # Remove prefixes
        expr = re.sub(r'^.*?BARRIER.*?:\s*', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'^.*?B\(.*?\)\s*=\s*', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'^.*?[:=]\s*', '', expr)
        
        # Fix unicode math symbols
        expr = expr.replace('²', '**2')
        expr = expr.replace('³', '**3')
        expr = expr.replace('⁴', '**4')
        expr = expr.replace('⁰', '**0')
        expr = expr.replace('¹', '**1')
        expr = expr.replace('⁵', '**5')
        expr = expr.replace('⁶', '**6')
        expr = expr.replace('⁷', '**7')
        expr = expr.replace('⁸', '**8')
        expr = expr.replace('⁹', '**9')
        
        # Convert ^ to **
        expr = re.sub(r'\^(\d+)', r'**\1', expr)
        expr = expr.replace('^', '**')
        
        # Fix common math notation issues
        expr = re.sub(r'(\w)(\()', r'\1*\2', expr)  # x(something) -> x*(something)
        expr = re.sub(r'(\))(\w)', r'\1*\2', expr)  # (something)x -> (something)*x
        
        # Fix parentheses around numbers
        expr = re.sub(r'\((-?\d+\.?\d*)\)', r'\1', expr)
        
        # Fix missing multiplication between number and variable
        expr = re.sub(r'(\d)([x])', r'\1*\2', expr)
        expr = re.sub(r'(\d)\s+([x])', r'\1*\2', expr)
        
        # Fix spaces in expressions
        expr = re.sub(r'\s+', ' ', expr).strip()
        expr = expr.rstrip('.,;:')
        
        # Fix multiple consecutive operators
        expr = re.sub(r'\*\*+', '**', expr)
        expr = re.sub(r'\++', '+', expr)
        expr = re.sub(r'--+', '-', expr)
        
        # Remove invalid characters
        expr = re.sub(r'[^\w\s\+\-\*\(\)\.\/]', '', expr)
        
        return expr

    def _validate_expression(self, expr: str) -> bool:
        """Validate expression more thoroughly"""
        if not expr or len(expr) < 3:
            return False
        
        # Must contain variables
        if not re.search(r'x\d+', expr):
            return False
        
        # Must have numerical coefficients
        if not re.search(r'\d', expr):
            return False
        
        # Check for valid Python syntax by trying to compile
        try:
            # Replace variables with dummy values for syntax check
            test_expr = expr
            for i in range(1, 10):  # Check x1 through x9
                test_expr = test_expr.replace(f'x{i}', '1')
            
            # Try to compile as Python expression
            compile(test_expr, '<string>', 'eval')
            return True
            
        except SyntaxError:
            logger.warning(f"Invalid Python syntax in expression: {expr}")
            return False
        except Exception as e:
            logger.warning(f"Expression validation error: {e}")
            return False

    def _extract_barrier_and_controller_from_response(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract both barrier and controller from LLM response"""
        # Try structured patterns first
        barrier_patterns = [
            r'BARRIER\s*:\s*(.+?)(?:\n|$)',
            r'BARRIER_CERTIFICATE\s*:\s*(.+?)(?:\n|$)',
            r'B\(x\)\s*=\s*(.+?)(?:\n|$)'
        ]
        
        controller_patterns = [
            r'CONTROLLER\s*:\s*(.+?)(?:\n|$)',
            r'CONTROL\s*:\s*(.+?)(?:\n|$)',
            r'u\s*=\s*(.+?)(?:\n|$)'
        ]
        
        barrier_expr = None
        controller_expr = None
        
        # Extract barrier
        for pattern in barrier_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                expr = matches[0].strip()
                cleaned = self._clean_expression(expr)
                if cleaned and self._validate_expression(cleaned):
                    barrier_expr = cleaned
                    break
        
        # Extract controller
        for pattern in controller_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                expr = matches[0].strip()
                cleaned = self._clean_controller_expression(expr)
                if cleaned:
                    controller_expr = cleaned
                    break
        
        return barrier_expr, controller_expr

    def _extract_refined_barrier_and_controller_from_response(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract refined barrier and controller from LLM response"""
        # Try structured patterns first
        barrier_patterns = [
            r'REFINED_BARRIER\s*:\s*(.+?)(?:\n|$)',
            r'BARRIER\s*:\s*(.+?)(?:\n|$)',
        ]
        
        controller_patterns = [
            r'REFINED_CONTROLLER\s*:\s*(.+?)(?:\n|$)',
            r'CONTROLLER\s*:\s*(.+?)(?:\n|$)',
        ]
        
        barrier_expr = None
        controller_expr = None
        
        # Extract barrier
        for pattern in barrier_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                expr = matches[0].strip()
                cleaned = self._clean_expression(expr)
                if cleaned and self._validate_expression(cleaned):
                    barrier_expr = cleaned
                    break
        
        # Extract controller
        for pattern in controller_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                expr = matches[0].strip()
                cleaned = self._clean_controller_expression(expr)
                if cleaned:
                    controller_expr = cleaned
                    break
        
        return barrier_expr, controller_expr

    def _clean_controller_expression(self, expr: str) -> str:
        """Clean controller expression"""
        if not expr:
            return ""
        
        # Basic cleaning
        expr = expr.replace('`', '').strip()
        
        # Remove prefixes
        expr = re.sub(r'^.*?CONTROLLER.*?:\s*', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'^.*?CONTROL.*?:\s*', '', expr, flags=re.IGNORECASE)
        
        # Fix unicode math symbols
        expr = expr.replace('²', '**2')
        expr = expr.replace('³', '**3')
        expr = re.sub(r'\^(\d+)', r'**\1', expr)
        expr = expr.replace('^', '**')
        
        # Fix missing multiplication between number and variable
        expr = re.sub(r'(\d)([x])', r'\1*\2', expr)
        expr = re.sub(r'(\d)\s+([x])', r'\1*\2', expr)
        
        # Clean up spaces
        expr = re.sub(r'\s+', ' ', expr).strip()
        expr = expr.rstrip('.,;:')
        
        return expr

    def _parse_controller_expressions(self, controller_expr: str, problem: Dict[str, Any]) -> Dict[str, str]:
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
            logger.error(f"Failed to parse controller expressions: {e}")
            return {}

    def _substitute_controller_into_dynamics(self, dynamics_str: str, controller_dict: Dict[str, str]) -> str:
        """Substitute controller expressions into dynamics string to create closed-loop dynamics"""
        try:
            if not controller_dict:
                logger.warning("Empty controller dictionary, returning original dynamics")
                return dynamics_str
                
            # Split dynamics into individual equations
            equations = [eq.strip() for eq in dynamics_str.split(',')]
            
            substituted_equations = []
            
            for eq in equations:
                eq = eq.strip()
                
                # Substitute each controller parameter
                for param_name, param_expr in controller_dict.items():
                    # Use word boundaries to avoid partial matches
                    pattern = r'\b' + re.escape(param_name) + r'\b'
                    
                    # Wrap controller expression in parentheses for safety
                    replacement = f'({param_expr})'
                    
                    eq = re.sub(pattern, replacement, eq)
                
                substituted_equations.append(eq)
            
            # Join back into single dynamics string
            closed_loop_dynamics = ', '.join(substituted_equations)
            
            logger.info(f"Original dynamics: {dynamics_str}")
            logger.info(f"Controller: {controller_dict}")
            logger.info(f"Closed-loop dynamics: {closed_loop_dynamics}")
            
            return closed_loop_dynamics
            
        except Exception as e:
            logger.error(f"Failed to substitute controller into dynamics: {e}")
            return dynamics_str  # Return original as fallback

    def _add_to_history(self, iteration: int, barrier: str, score: int, verification=None, controller: str = None):
        """Add to iteration history with controller support"""
        entry = {
            'iteration': iteration,
            'barrier': barrier,
            'score': score,
            'verification': verification
        }
        
        if controller is not None:
            entry['controller'] = controller
        
        self.iteration_history.append(entry)


# Test
if __name__ == "__main__":
    
    # test_problem = {
    #     'dynamics': 'dx1/dt = x2 + u0, dx2/dt = -0.3*x1 - 0.2*(1 + cos(x1))*x2 - 0.1*x1**3 + u1',
    #     'initial_set': {
    #         'type': 'bounds',
    #         'bounds': [[-0.5, 0.5], [-0.6, 0.6]]
    #     },
    #     'unsafe_set': {
    #         'type': 'ball',
    #         'radius': 1.0,
    #         'center': [2, 2],
    #         'complement': False
    #     },
    #     'controller_parameters': 'u0, u1',
    #     'description': "State-dependent trigonometric damping"
    # }

    test_problem = {
        "dynamics": "dx1/dt = -0.4*x1 + 0.1*x2, dx2/dt = -0.5*x2 + 0.08*x3, dx3/dt = -0.6*x3 + 0.05*x1, dx4/dt = -0.3*x4",
        "initial_set": {
          "type": "ball",
          "radius": 0.7,
          "center": [0, 0, 0, 0]
        },
        "unsafe_set": {
          "type": "ball",
          "radius": 2.5,
          "center": [0, 0, 0, 0],
          "complement": True
        },
        "description": "4D with weak coupling between x1-x2-x3 chain"
      }
    
    api_key = "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"
    
    synthesizer = SimplifiedBarrierSynthesis(
        api_key=api_key,
        max_iterations=5,
        dataset_json_path="barrier_dataset.json"
    )
    
    result = synthesizer.synthesize_barrier_certificate(test_problem)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if result['success']:
        print(f"✅ SUCCESS: {result['barrier_certificate']}")
        print(f"Found in iteration: {result['iteration_found']}")
    else:
        print(f"❌ FAILED - Best score: {result.get('best_score', 0)}/3")
        if result.get('best_barrier'):
            print(f"Best attempt: {result['best_barrier']}")
    
    print(f"Time: {result['total_time']:.2f}s")