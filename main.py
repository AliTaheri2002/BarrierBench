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
                                      template_type: str):
        """Save successful solution"""
        try:
            new_record = {
                'problem': problem,
                'barrier': barrier_certificate,
                'template_type': template_type
            }
            self.test_cases.append(new_record)
            self._save_dataset_to_json()
            logger.info(f"Saved to dataset: {barrier_certificate}")
        except Exception as e:
            logger.error(f"Failed to save: {e}")

    def find_most_similar(self, target_problem: Dict[str, Any], threshold: float = 0.7) -> Tuple[Optional[Dict], float]:
        """Find most similar problems above threshold and let LLM choose the best"""
        # Find all similar cases above threshold
        similar_cases = []
        for case in self.test_cases:
            similarity = self._calculate_similarity(target_problem, case['problem'])
            if similarity >= threshold:
                similar_cases.append((case, similarity))
        
        # Print similarity search results
        print(f"\n Similarity Search: Found {len(similar_cases)} cases above threshold {threshold}")
        if similar_cases:
            print("Top candidates:")
            for i, (case, sim) in enumerate(similar_cases[:5], 1):
                dynamics_short = case['problem'].get('dynamics', '')[:50] + "..." if len(case['problem'].get('dynamics', '')) > 50 else case['problem'].get('dynamics', '')
                print(f"  {i}. Score: {sim:.3f} - {dynamics_short}")
        
        if not similar_cases:
            print("No similar cases found")
            return None, 0.0
        
        # Sort by similarity (highest first)
        similar_cases.sort(key=lambda x: x[1], reverse=True)
        
        # If only one case, return it
        if len(similar_cases) == 1:
            print("Single match - using directly")
            return similar_cases[0][0], similar_cases[0][1]
        
        # If multiple cases, let LLM choose the best one
        print(f"Multiple matches - sending top 5 to LLM for selection")
        best_case = self._llm_select_best_similar_case(target_problem, similar_cases[:5])  # Top 5 cases
        
        if best_case:
            # Find similarity score for the selected case
            selected_similarity = next(sim for case, sim in similar_cases if case == best_case)
            print(f"LLM selected case with similarity: {selected_similarity:.3f}")
            return best_case, selected_similarity
        else:
            # Fallback to highest similarity
            print("LLM selection failed - using highest similarity")
            return similar_cases[0][0], similar_cases[0][1]

    def _llm_select_best_similar_case(self, target_problem: Dict[str, Any], 
                                     similar_cases: List[Tuple[Dict, float]]) -> Optional[Dict]:
        """Use LLM to select the best similar case from candidates"""
        try:
            # Create prompt for LLM to choose best case
            cases_description = ""
            for i, (case, similarity) in enumerate(similar_cases, 1):
                cases_description += f"""
CASE {i} (Similarity: {similarity:.3f}):
- Dynamics: {case['problem'].get('dynamics')}
- Initial set: {case['problem'].get('initial_set')}
- Unsafe set: {case['problem'].get('unsafe_set')}
- Successful barrier: {case['barrier']}
"""
            
            prompt = f"""TARGET PROBLEM:
- Dynamics: {target_problem.get('dynamics')}
- Initial set: {target_problem.get('initial_set')}
- Unsafe set: {target_problem.get('unsafe_set')}

SIMILAR CASES FOUND:{cases_description}

Which case is most structurally similar to the TARGET PROBLEM for barrier certificate design?
Consider: dynamics structure, system dimensions, set geometries, and mathematical complexity.

Answer with only the case number (1, 2, 3, 4, or 5): """

            # This would need the LLM client - for now return the highest similarity case
            # In the actual implementation, you would use self.client here
            return similar_cases[0][0]  # Fallback to highest similarity
            
        except Exception as e:
            logger.warning(f"LLM selection failed: {e}")
            return similar_cases[0][0]  # Fallback to highest similarity

    def _calculate_similarity(self, prob1: Dict[str, Any], prob2: Dict[str, Any]) -> float:
        """Advanced similarity calculation based on multiple mathematical features"""
        score = 0.0
        
        # 1. Dynamics Analysis (40% weight)
        dynamics_score = self._calculate_dynamics_similarity(
            prob1.get('dynamics', ''), 
            prob2.get('dynamics', '')
        )
        score += 0.4 * dynamics_score
        
        # 2. Set Geometry Similarity (35% weight)
        set_score = self._calculate_set_similarity(
            prob1.get('initial_set', {}), prob1.get('unsafe_set', {}),
            prob2.get('initial_set', {}), prob2.get('unsafe_set', {})
        )
        score += 0.35 * set_score
        
        # 3. System Structure Similarity (25% weight)
        structure_score = self._calculate_structure_similarity(prob1, prob2)
        score += 0.25 * structure_score
        
        return min(score, 1.0)  # Cap at 1.0

    def _calculate_dynamics_similarity(self, dyn1: str, dyn2: str) -> float:
        """Calculate similarity between dynamics expressions"""
        if not dyn1 or not dyn2:
            return 0.0
        
        score = 0.0
        
        # Extract mathematical features
        features1 = self._extract_dynamics_features(dyn1)
        features2 = self._extract_dynamics_features(dyn2)
        
        # Linearity comparison (30%)
        if features1['is_linear'] == features2['is_linear']:
            score += 0.3
        
        # Polynomial degree comparison (25%)
        degree_diff = abs(features1['max_degree'] - features2['max_degree'])
        if degree_diff == 0:
            score += 0.25
        elif degree_diff == 1:
            score += 0.15
        
        # Coupling strength comparison (20%)
        coupling_diff = abs(features1['coupling_strength'] - features2['coupling_strength'])
        if coupling_diff < 0.3:
            score += 0.2 * (1 - coupling_diff / 0.3)
        
        # Variable count comparison (15%)
        if features1['var_count'] == features2['var_count']:
            score += 0.15
        
        # System type comparison (10%)
        if features1['system_type'] == features2['system_type']:
            score += 0.1
        
        return score

    def _extract_dynamics_features(self, dynamics: str) -> Dict[str, Any]:
        """Extract mathematical features from dynamics string"""
        features = {
            'is_linear': True,
            'max_degree': 1,
            'coupling_strength': 0.0,
            'var_count': 0,
            'system_type': 'unknown',
            'has_constants': False,
            'cross_terms': False
        }
        
        # Count variables
        variables = set(re.findall(r'x\d+', dynamics))
        features['var_count'] = len(variables)
        
        # Check linearity and degree
        # Look for polynomial terms like x1**2, x1*x2, etc.
        nonlinear_patterns = [
            r'x\d+\*\*\d+',  # x1**2, x2**3, etc.
            r'x\d+\*x\d+',   # x1*x2, etc.
        ]
        
        max_degree = 1
        has_cross_terms = False
        
        for pattern in nonlinear_patterns:
            matches = re.findall(pattern, dynamics)
            if matches:
                features['is_linear'] = False
                for match in matches:
                    if '**' in match:
                        degree = int(re.search(r'\*\*(\d+)', match).group(1))
                        max_degree = max(max_degree, degree)
                    elif '*' in match and 'x' in match:
                        # Cross term like x1*x2
                        has_cross_terms = True
                        max_degree = max(max_degree, 2)
        
        features['max_degree'] = max_degree
        features['cross_terms'] = has_cross_terms
        
        # Estimate coupling strength (how much variables influence each other)
        coupling_count = len(re.findall(r'x\d+.*x\d+', dynamics))
        total_terms = len(re.findall(r'[+-]', dynamics)) + 1
        features['coupling_strength'] = coupling_count / max(total_terms, 1)
        
        # Check for constants
        features['has_constants'] = bool(re.search(r'\d+\.?\d*', dynamics))
        
        # Classify system type
        if features['is_linear']:
            if features['coupling_strength'] > 0.3:
                features['system_type'] = 'coupled_linear'
            else:
                features['system_type'] = 'decoupled_linear'
        else:
            if features['max_degree'] == 2:
                features['system_type'] = 'quadratic'
            elif features['max_degree'] > 2:
                features['system_type'] = 'polynomial_high'
            else:
                features['system_type'] = 'nonlinear_other'
        
        return features

    def _calculate_set_similarity(self, init1: Dict, unsafe1: Dict, 
                                 init2: Dict, unsafe2: Dict) -> float:
        """Calculate similarity between initial and unsafe sets"""
        score = 0.0
        
        # Initial set comparison (50%)
        init_score = self._compare_sets(init1, init2)
        score += 0.5 * init_score
        
        # Unsafe set comparison (50%)
        unsafe_score = self._compare_sets(unsafe1, unsafe2)
        score += 0.5 * unsafe_score
        
        return score

    def _compare_sets(self, set1: Dict, set2: Dict) -> float:
        """Compare two set definitions"""
        if not set1 or not set2:
            return 0.0
        
        score = 0.0
        
        # Type similarity (40%)
        if set1.get('type') == set2.get('type'):
            score += 0.4
        
        # Dimension similarity (20%)
        dim1 = len(set1.get('center', [0, 0]))
        dim2 = len(set2.get('center', [0, 0]))
        if dim1 == dim2:
            score += 0.2
        
        # Complement similarity (20%)
        comp1 = set1.get('complement', False)
        comp2 = set2.get('complement', False)
        if comp1 == comp2:
            score += 0.2
        
        # Size similarity (20%) - for balls and boxes
        if set1.get('type') == set2.get('type') == 'ball':
            r1 = set1.get('radius', 1.0)
            r2 = set2.get('radius', 1.0)
            ratio = min(r1, r2) / max(r1, r2)
            score += 0.2 * ratio
        elif set1.get('type') == set2.get('type') == 'box':
            size1 = set1.get('size', [1, 1])
            size2 = set2.get('size', [1, 1])
            if len(size1) == len(size2):
                ratios = [min(s1, s2) / max(s1, s2) for s1, s2 in zip(size1, size2)]
                avg_ratio = sum(ratios) / len(ratios)
                score += 0.2 * avg_ratio
        
        return score

    def _calculate_structure_similarity(self, prob1: Dict[str, Any], prob2: Dict[str, Any]) -> float:
        """Calculate overall structural similarity"""
        score = 0.0
        
        # Problem complexity similarity (50%)
        complexity1 = self._estimate_problem_complexity(prob1)
        complexity2 = self._estimate_problem_complexity(prob2)
        complexity_diff = abs(complexity1 - complexity2)
        if complexity_diff < 0.2:
            score += 0.5 * (1 - complexity_diff / 0.2)
        
        # Scale similarity (30%)
        scale1 = self._estimate_problem_scale(prob1)
        scale2 = self._estimate_problem_scale(prob2)
        if scale1 and scale2:
            scale_ratio = min(scale1, scale2) / max(scale1, scale2)
            score += 0.3 * scale_ratio
        
        # Symmetry similarity (20%)
        sym1 = self._check_symmetry(prob1)
        sym2 = self._check_symmetry(prob2)
        if sym1 == sym2:
            score += 0.2
        
        return score

    def _estimate_problem_complexity(self, problem: Dict[str, Any]) -> float:
        """Estimate problem complexity (0-1 scale)"""
        complexity = 0.0
        
        dynamics = problem.get('dynamics', '')
        
        # Base complexity from dynamics
        if 'x1**' in dynamics or 'x2**' in dynamics:
            complexity += 0.4  # Nonlinear
        else:
            complexity += 0.2  # Linear
        
        # Additional complexity from cross terms
        if re.search(r'x\d+.*x\d+', dynamics):
            complexity += 0.3
        
        # Complexity from set types
        init_type = problem.get('initial_set', {}).get('type', '')
        unsafe_type = problem.get('unsafe_set', {}).get('type', '')
        
        if init_type in ['ellipse', 'polygon']:
            complexity += 0.15
        if unsafe_type in ['ellipse', 'polygon']:
            complexity += 0.15
        
        return min(complexity, 1.0)

    def _estimate_problem_scale(self, problem: Dict[str, Any]) -> Optional[float]:
        """Estimate characteristic length scale of problem"""
        try:
            init_set = problem.get('initial_set', {})
            unsafe_set = problem.get('unsafe_set', {})
            
            scales = []
            
            if init_set.get('type') == 'ball':
                scales.append(init_set.get('radius', 1.0))
            elif init_set.get('type') == 'box':
                size = init_set.get('size', [1, 1])
                scales.append(max(size))
            
            if unsafe_set.get('type') == 'ball':
                scales.append(unsafe_set.get('radius', 1.0))
            elif unsafe_set.get('type') == 'box':
                size = unsafe_set.get('size', [1, 1])
                scales.append(max(size))
            
            return max(scales) if scales else None
            
        except:
            return None

    def _check_symmetry(self, problem: Dict[str, Any]) -> str:
        """Check for symmetries in the problem"""
        dynamics = problem.get('dynamics', '')
        init_center = problem.get('initial_set', {}).get('center', [0, 0])
        unsafe_center = problem.get('unsafe_set', {}).get('center', [0, 0])
        
        # Check if centered at origin
        origin_centered = (all(abs(c) < 1e-6 for c in init_center) and 
                          all(abs(c) < 1e-6 for c in unsafe_center))
        
        # Check for rotational symmetry in dynamics
        # This is a simple check - could be made more sophisticated
        if origin_centered and 'x1*x2' not in dynamics:
            return 'radial'
        elif origin_centered:
            return 'origin'
        else:
            return 'none'

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
        """Main synthesis pipeline"""
        logger.info("Starting Simplified Barrier Synthesis")
        start_time = time.time()
        self.iteration_history = []
        
        try:
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"\n=== ITERATION {iteration}/{self.max_iterations} ===")
                
                # Step 1: Get template from LLM
                template_and_barrier = self._get_template_and_barrier_from_llm(problem, iteration)
                
                if not template_and_barrier:
                    logger.warning(f"Failed to get template/barrier in iteration {iteration}")
                    continue
                
                barrier_expr = template_and_barrier
                logger.info(f"LLM suggested: {barrier_expr}")
                
                # Step 2: Verify
                verification_result = self._verify_barrier(barrier_expr, problem)
                
                if not verification_result:
                    logger.warning(f"Verification failed in iteration {iteration}")
                    self._add_to_history(iteration, barrier_expr, 0, "verification_error")
                    continue
                
                score = sum([
                    verification_result['condition_1'],
                    verification_result['condition_2'], 
                    verification_result['condition_3']
                ])
                
                logger.info(f"Initial score: {score}/3")
                
                # Step 3: Refinements (up to 4)
                original_barrier = barrier_expr  # Keep original barrier for refinement context
                best_barrier = barrier_expr
                best_score = score
                best_verification = verification_result
                
                # Track refinements for this iteration
                iteration_refinements = []
                
                if score < 3:
                    for refinement in range(1, 5):  # 4 refinements max
                        logger.info(f"Refinement {refinement}/4")
                        
                        # Prepare refinement context
                        refined = self._refine_barrier(original_barrier, best_barrier, problem, best_verification, refinement, iteration_refinements)
                        
                        if not refined:
                            continue
                        
                        ref_verification = self._verify_barrier(refined, problem)
                        if not ref_verification:
                            continue
                        
                        # Store this refinement with its verification details (AFTER verification)
                        iteration_refinements.append({
                            'refinement_num': refinement,
                            'barrier': refined,
                            'base_barrier': best_barrier,
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
                            best_barrier = refined
                            best_score = ref_score
                            best_verification = ref_verification
                        
                        if ref_score == 3:
                            break
                
                # Store result (this is the FINAL best barrier for this iteration - could be original or refined)
                self._add_to_history(iteration, best_barrier, best_score, best_verification)
                
                # Success check
                if best_score == 3:
                    elapsed_time = time.time() - start_time
                    logger.info(f"SUCCESS! Found valid barrier in iteration {iteration}")
                    
                    # Save to dataset
                    try:
                        self.dataset.save_successful_barrier_to_json(
                            problem=problem,
                            barrier_certificate=best_barrier,
                            template_type="llm_generated"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save to dataset: {e}")
                    
                    return {
                        'success': True,
                        'barrier_certificate': best_barrier,
                        'iteration_found': iteration,
                        'total_time': elapsed_time,
                        'verification_details': best_verification
                    }
            
            # No success
            elapsed_time = time.time() - start_time
            best_attempt = max(self.iteration_history, key=lambda x: x['score']) if self.iteration_history else None
            
            return {
                'success': False,
                'best_score': best_attempt['score'] if best_attempt else 0,
                'best_barrier': best_attempt['barrier'] if best_attempt else None,
                'total_time': elapsed_time,
                'total_iterations': len(self.iteration_history)
            }
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time
            }

    def _get_template_and_barrier_from_llm(self, problem: Dict[str, Any], iteration: int) -> Optional[str]:
        """Get template and barrier from LLM"""
        
        # Check for similar problems
        if iteration == 1:
            similar_case, similarity = self.dataset.find_most_similar(problem)
            context = ""
            if similar_case and similarity > 0.99:  # Changed from 0.5 to 0.6
                context = f"""Related problem found:
EXAMPLE: {similar_case['problem']}
B(x): {similar_case['barrier']}

WARNING: This is just an example - your solution may be completely different NOT ONLY in terms of coefficients, BUT ALSO in format and structure. you may make mistakes, so do not consider this as a solution. Please don't copy it

"""
            else:
                context = "No similar problems found. Analyze this problem fresh.\n\n"
        else:
            # Provide previous failures
            context = self._prepare_iteration_context(iteration)
        
        prompt = f"""{context}Main Problem:
- Dynamics: {problem.get('dynamics')}
- Initial set: {problem.get('initial_set')}
- Unsafe set: {problem.get('unsafe_set')}

Design a barrier certificate B(x) that satisfies:
1. B(x) ≤ 0 in initial set
2. B(x) > 0 in unsafe set  
3. ∇B·f < 0 on boundary

{"Design a barrier certificate that satisfies all 3 conditions. Be very careful - don't make it more complex than needed." if iteration == 1 else "Design a barrier certificate that satisfies all 3 conditions. Learn from previous failures. You can change structure of TEMPLATE if needed. In this step, the goal is to improve the structure of the templates, not refine the parameters."}

CRITICAL: Use ONLY real numbers in the barrier expression. No variables like 'c' or 'ε'. 
Solve specifically for THIS problem with appropriate coefficients.

Analyze carefully but be concise. Give precise answer without long explanations.

BARRIER: [expression with numbers only]"""

        print("\n" + "="*80)
        print(f"TEMPLATE/BARRIER PROMPT (Iteration {iteration}):")
        print("="*80)
        print(prompt)
        print("="*80)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1200,  # Increased from 800 to 1200 (1.5x)
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            print("\n" + "="*80)
            print(f"CLAUDE TEMPLATE/BARRIER RESPONSE:")
            print("="*80)
            print(content)
            print("="*80)
            
            barrier_expr = self._extract_barrier_from_response(content)
            
            if barrier_expr:
                logger.info(f"Successfully extracted barrier: {barrier_expr}")
            else:
                logger.warning("Failed to extract barrier from response")
            
            return barrier_expr
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return None

    def _prepare_iteration_context(self, iteration: int) -> str:
        """Prepare context from previous failures"""
        if not self.iteration_history:
            return ""
        
        # Get FINAL best barriers from each previous iteration (could be original or refined)
        context = "Previous attempts failed:\n"
        for hist in self.iteration_history[-3:]:  # Last 3 iterations max
            failed_conditions = self._get_failed_conditions(hist.get('verification', {}))
            # hist['barrier'] is the FINAL best barrier from that iteration (original or best refinement)
            context += f"- Tried: {hist['barrier']} (failed: {failed_conditions})\n"
        
        # Add general guidance
        context += "\nImprove the barrier structure to satisfy all conditions.\n\n"
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
                       iteration_refinements: List[Dict] = None) -> Optional[str]:
        """Refine barrier based on failures"""
        
        failed_conditions = self._get_failed_conditions(verification)
        
        # Prepare refinement context
        if refinement_num == 1:
            # For first refinement, show original barrier with its failed conditions
            original_failed = self._get_failed_conditions(verification)
            refinement_context = f"Original barrier: {original_barrier} (failed: {original_failed})\n"
        else:
            # For refinements 2-4, show original and all previous refinements with their failed conditions
            original_failed = self._get_failed_conditions(verification)
            refinement_context = f"Original barrier: {original_barrier} (failed: {original_failed})\n"
            
            if iteration_refinements:
                for i, ref_data in enumerate(iteration_refinements, 1):
                    ref_barrier = ref_data['barrier']
                    ref_failed = ref_data['failed_conditions']
                    refinement_context += f"Refinement {i}: {ref_barrier} (failed: {ref_failed})\n"
        
        # Different instructions based on refinement number
        if refinement_num <= 2:
            instruction = "Try a different coefficient distribution. You can redistribute the coefficients between ALL terms (sometimes it is necessary for all terms to have different coefficients), but DO NOT change structure"
        else:
            instruction = "Previous coefficient adjustments failed. Consider changing the barrier structure if needed while keeping the same polynomial degree. You can modify the terms or their combinations."
        
        prompt = f"""{refinement_context}
This barrier violates: {failed_conditions}

Problem:
- Dynamics: {problem.get('dynamics')}
- Initial set: {problem.get('initial_set')}
- Unsafe set: {problem.get('unsafe_set')}

{instruction}

Requirements:
1. B(x) ≤ 0 in initial set
2. B(x) > 0 in unsafe set  
3. ∇B(x)·f(x) < 0 on boundary

Analyze carefully but be concise. Give precise answer without long explanations.

REFINED_BARRIER: [expression with numbers only]"""

        print("\n" + "="*80)
        print(f"REFINEMENT PROMPT (Attempt {refinement_num}):")
        print("="*80)
        print(prompt)
        print("="*80)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=900,  # Increased from 600 to 900 (1.5x)
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            print("\n" + "="*80)
            print(f"CLAUDE REFINEMENT RESPONSE (Attempt {refinement_num}):")
            print("="*80)
            print(content)
            print("="*80)
            
            refined_expr = self._extract_barrier_from_response(content)
            
            if refined_expr:
                logger.info(f"Refinement {refinement_num} produced: {refined_expr}")
            else:
                logger.warning(f"Failed to extract refined barrier from refinement {refinement_num}")
            
            return refined_expr
            
        except Exception as e:
            logger.error(f"Refinement error: {e}")
            return None

    def _verify_barrier(self, barrier_expr: str, problem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Verify barrier with sample gatekeeper + SMT"""
        try:
            # Check dimensions
            system_vars = self._extract_system_variables(problem.get('dynamics', ''))
            barrier_vars = self._extract_barrier_variables(barrier_expr)
            
            missing_vars = set(system_vars) - set(barrier_vars)
            if missing_vars:
                logger.error(f"Missing variables: {missing_vars}")
                return None

            # Sample gatekeeper
            try:
                from data_structures import generate_samples_for_barrier_validation, validate_barrier_on_samples
                
                samples = generate_samples_for_barrier_validation(problem, num_samples=5000)
                sample_validation = validate_barrier_on_samples(barrier_expr, problem, samples)
                
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

            # SMT verification
            smt_validation = get_detailed_condition_results(
                barrier_expr,
                problem['initial_set'],
                problem['unsafe_set'], 
                problem['dynamics']
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

    def _add_to_history(self, iteration: int, barrier: str, score: int, verification=None):
        """Add to iteration history"""
        self.iteration_history.append({
            'iteration': iteration,
            'barrier': barrier,
            'score': score,
            'verification': verification
        })


# Test
if __name__ == "__main__":
    test_problem = {
        'dynamics': 'x1[k+1] = 0.9*x1[k], x2[k+1] = 0.8*x2[k]',
        'initial_set': {
            'type': 'ball',
            'radius': 1.0,
            'center': [0, 0]
        },
        'unsafe_set': {
            'type': 'ball',
            'radius': 3.0,
            'center': [0, 0],
            'complement': True
        },
        'description': "Discrete Linear System"
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
        print(f"SUCCESS: {result['barrier_certificate']}")
        print(f"Found in iteration: {result['iteration_found']}")
    else:
        print(f"FAILED - Best score: {result.get('best_score', 0)}/3")
        if result.get('best_barrier'):
            print(f"Best attempt: {result['best_barrier']}")
    
    print(f"Time: {result['total_time']:.2f}s")