import os
import sys
import logging
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from copy import deepcopy
import anthropic
import re

# Import our modules
from data_structures import (
    BarrierSolution, 
    ProblemAnalysis, 
    generate_samples_for_barrier_validation, 
    validate_barrier_on_samples
)
from barrier_verification import get_detailed_condition_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGoTBarrierSynthesis:
    """
    Enhanced Graph of Thoughts based Barrier Certificate Synthesis Framework
    
    Key Features:
    - Supports systems with up to 10 variables (x1 to x10)
    - Supports 4th power terms and higher
    - Processes 4 templates simultaneously per iteration
    - Parallel prediction and refinement for each template
    - Aggregation in every iteration (including first)
    - Template evolution across iterations
    - Early stopping when perfect solution found
    """
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", 
                 max_iterations: int = 7, max_refinements: int = 3):
        """
        Initialize the Enhanced GoT Barrier Synthesis framework
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.max_refinements = max_refinements
        
        # Storage for best solution per template type per iteration
        self.current_iteration_solutions: Dict[int, BarrierSolution] = {}  # template_index -> best_solution
        self.all_template_history: List[List[str]] = []  # Track all templates used across iterations
        self.analysis_history: List[ProblemAnalysis] = []
        
        # Current best solution overall
        self.best_solution: Optional[BarrierSolution] = None
        
        logger.info(f"Initialized Enhanced GoT Barrier Synthesis with 4 templates per iteration (supports x1-x10, 4th powers)")

    def synthesize_barrier_certificate(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for enhanced barrier certificate synthesis
        """
        logger.info("Starting Enhanced GoT Barrier Certificate Synthesis (4 Templates Parallel, x1-x10 Support)")
        logger.info(f"Problem: {json.dumps(problem, indent=2)}")
        
        start_time = time.time()
        
        # Reset state for new problem
        self._reset_state()
        
        try:
            # Main iteration loop
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"ITERATION {iteration}/{self.max_iterations}")
                logger.info(f"{'='*60}")
                
                # Run iteration with 4 templates
                success = self._run_iteration_multi_template(problem, iteration)
                
                if success:
                    logger.info(f"SUCCESS! Valid barrier certificate found in iteration {iteration}")
                    break
                    
                logger.info(f"Iteration {iteration} failed to find valid solution")
            
            # Prepare final results
            elapsed_time = time.time() - start_time
            
            if self.best_solution and self.best_solution.score == 3:
                return {
                    'success': True,
                    'barrier_certificate': self.best_solution.expression,
                    'template_type': self.best_solution.template_type,
                    'iteration_found': self.best_solution.iteration,
                    'phase_found': self.best_solution.phase,
                    'verification_details': self.best_solution.verification_details,
                    'total_time': elapsed_time,
                    'total_iterations': len(self.analysis_history),
                    'templates_explored': len([t for templates in self.all_template_history for t in templates])
                }
            else:
                return {
                    'success': False,
                    'best_solution': self._solution_to_dict(self.best_solution) if self.best_solution else None,
                    'total_time': elapsed_time,
                    'total_iterations': len(self.analysis_history),
                    'templates_explored': len([t for templates in self.all_template_history for t in templates]),
                    'message': 'No valid barrier certificate found within iteration limit'
                }
                
        except Exception as e:
            logger.error(f"Fatal error in synthesis: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time
            }

    def _generate_samples_for_barrier_validation(self, problem: Dict[str, Any], num_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate samples for barrier validation following k=0 paper methodology
        """
        return generate_samples_for_barrier_validation(problem, num_samples)

    def _validate_barrier_on_samples(self, barrier_expr: str, problem: Dict[str, Any], samples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate barrier certificate on samples using k=0 simple validation methodology
        """
        return validate_barrier_on_samples(barrier_expr, problem, samples)

    def _run_iteration_multi_template(self, problem: Dict[str, Any], iteration: int) -> bool:
        """
        Run a single iteration with 4 templates processed in parallel
        """
        # Phase 1: Multi-Template Analysis (4 templates)
        logger.info(f"Phase 1: Multi-Template Analysis (4 templates)")
        analysis = self._analyze_problem_multi_template(problem, iteration)
        self.analysis_history.append(analysis)
        
        # Store templates for this iteration
        self.all_template_history.append(analysis.suggested_templates)
        
        # Phase 2: Parallel Template Processing (4 solutions)
        logger.info(f"Phase 2: Parallel Template Processing")
        template_solutions = self._process_templates_parallel(analysis, problem, iteration)
        
        # Check if any perfect solution found
        for solution in template_solutions.values():
            if solution and solution.score == 3:
                self._update_best_solution(solution)
                logger.info(f"Perfect solution found in template {solution.template_index}!")
                return True
        
        # Phase 3: Aggregation (always run, even in iteration 1)
        logger.info(f"Phase 3: Multi-Template Aggregation")
        aggregated_solution = self._aggregate_multi_template_solutions(
            template_solutions, analysis, problem, iteration
        )
        
        if aggregated_solution and aggregated_solution.score == 3:
            self._update_best_solution(aggregated_solution)
            logger.info(f"Perfect solution found through aggregation!")
            return True
        
        # Update current iteration solutions for next iteration
        self.current_iteration_solutions = template_solutions
        
        # Update best solution if improved
        all_solutions = list(template_solutions.values())
        if aggregated_solution:
            all_solutions.append(aggregated_solution)
        
        for solution in all_solutions:
            if solution:
                self._update_best_solution(solution)
        
        return False

    def _analyze_problem_multi_template(self, problem: Dict[str, Any], iteration: int) -> ProblemAnalysis:
        """
        Analyze the problem and suggest 4 appropriate barrier templates - supports x1-x10
        """
        if iteration == 1:
            # First iteration: Use predefined 4 fundamental templates
            return self._get_first_iteration_templates(problem)
        else:
            # Later iterations: Analyze failures and suggest new templates
            return self._get_evolved_templates(problem, iteration)

    def _get_first_iteration_templates(self, problem: Dict[str, Any]) -> ProblemAnalysis:
        """
        Get the 4 predefined templates for first iteration - enhanced for x1-x10 and 4th powers
        """
        prompt = f"""You are an expert in barrier certificate synthesis for dynamical systems. 

Problem Definition:
- Dynamics: {problem.get('dynamics', 'Not specified')}
- Initial Set: {json.dumps(problem.get('initial_set', {}), indent=2)}
- Unsafe Set: {json.dumps(problem.get('unsafe_set', {}), indent=2)}

You can use 4 fundamental barrier certificate approaches. 

1. **Diagonal Quadratic Barrier (Multi-Variable)**
   - Mathematical Structure: Sum of squared terms across variables
   - Purpose: Axis-aligned elliptical boundaries, scalable to N variables
   - Example: a1*x1**2 + a2*x2**2 + a3*x3**2 + ... + a_constant

2. **Coupled Quadratic Barrier (Cross-Terms)** 
   - Mathematical Structure: Includes cross-coupling terms between variables
   - Purpose: Rotated elliptical boundaries with variable interactions
   - Example: a1*x1**2 + a2*x1*x2 + a3*x2**2 + a4*x1*x3 + a5*x3**2 + ... + a_constant

3. **Higher-Order Polynomial Barrier (4th Powers)**
   - Mathematical Structure: Includes 4th power terms for stronger boundaries
   - Purpose: Non-elliptical, more complex boundary shapes
   - Example: a1*x1**4 + a2*x2**4 + a3*x1**2*x2**2 + a4*x1**2 + a5*x2**2 + a6

4. **Mixed Linear-Quadratic Barrier**
   - Mathematical Structure: Combination of linear and quadratic terms
   - Purpose: Asymmetric, parabolic boundaries
   - Example: a1*x1**2 + a2*x1 + a3*x2 + a4*x3**2 + a5*x3 + a6

Your task is to analyze the problem characteristics and create 4 templates suitable for the specific problem dimensions and complexity.

**IMPORTANT GUIDELINES:**
- Templates should match the problem's variable dimension (use only relevant variables)
- Ensure proper coefficient labeling (a1, a2, etc.)
- Templates should be mathematically distinct and complementary

Please provide:
1. **Dynamics Analysis**: System behavior, stability, key characteristics  
2. **Set Geometry**: Spatial relationship between initial and unsafe sets
3. **Template Suitability**: Why each of the 4 types could work for this problem
4. **Mathematical Insights**: Key properties for barrier synthesis

Format your response as (Don't make it bold):
DYNAMICS_DESCRIPTION: [your analysis]
SET_DESCRIPTION: [your analysis]  
TEMPLATE_1: [exact algebraic template form]
TEMPLATE_1_REASONING: [why this template suits this problem]
TEMPLATE_2: [exact algebraic template form]
TEMPLATE_2_REASONING: [why this template suits this problem]
TEMPLATE_3: [exact algebraic template form with 4th powers if appropriate]
TEMPLATE_3_REASONING: [why this template suits this problem]
TEMPLATE_4: [exact algebraic template form]
TEMPLATE_4_REASONING: [why this template suits this problem]
MATHEMATICAL_INSIGHTS: [key insights for barrier synthesis]
"""
        
        print("\n" + "="*80)
        print("SENDING FIRST ITERATION MULTI-TEMPLATE ANALYSIS PROMPT (x1-x10 Support):")
        print("="*80)
        print(prompt)
        print("="*80)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text

        print("\n" + "="*80)
        print("CLAUDE FIRST ITERATION MULTI-TEMPLATE RESPONSE:")
        print("="*80)
        print(content)
        print("="*80)
        
        # Parse response and ensure we have the 4 predefined templates
        analysis = self._parse_multi_template_analysis(content)
        
        # If parsing failed, use enhanced predefined templates as fallback
        if not analysis.suggested_templates or len(analysis.suggested_templates) < 4:
            logger.warning("Using fallback templates with x1-x10 and 4th power support")
            
            # Enhanced predefined templates for higher dimensions and 4th powers
            predefined_templates = [
                "a1*x1**2 + a2*x2**2 + a3*x3**2 + a4",                           # Diagonal quadratic (N-dimensional)
                "a1*x1**2 + a2*x1*x2 + a3*x2**2 + a4*x1*x3 + a5*x3**2 + a6",    # Coupled quadratic (cross-terms)
                "a1*x1**4 + a2*x2**4 + a3*x1**2*x2**2 + a4*x1**2 + a5*x2**2 + a6", # 4th power polynomial
                "a1*x1**2 + a2*x1 + a3*x2 + a4*x3**2 + a5*x3 + a6"              # Mixed linear-quadratic
            ]
            
            predefined_reasoning = [
                "Diagonal quadratic: Elliptical barriers aligned with coordinate axes, scalable to N variables",
                "Coupled quadratic: General elliptical barriers with rotation via cross-coupling terms between variables", 
                "Higher-order polynomial: 4th power terms enable complex, non-elliptical boundary shapes for challenging geometries",
                "Mixed linear-quadratic: Parabolic boundaries with asymmetric, open-ended separation capabilities"
            ]
            
            analysis.suggested_templates = predefined_templates
            analysis.template_reasoning = predefined_reasoning
        
        logger.info(f"First iteration analysis complete with 4 templates supporting x1-x10 and 4th powers")
        return analysis

    def _get_evolved_templates(self, problem: Dict[str, Any], iteration: int) -> ProblemAnalysis:
        """
        Get evolved templates for iterations > 1 based on previous failures - supports x1-x10 and 4th powers
        """
        # Prepare previous template information
        all_previous = [t for templates in self.all_template_history for t in templates]
        previous_templates_info = f"""
Previous iterations used these templates (which were insufficient):
{chr(10).join([f"- {template}" for template in all_previous])}

Please analyze why these templates failed and suggest 4 DIFFERENT, better templates. Learn from past failure experiences and be creative with NEW mathematical forms.
"""

        prompt = f"""You are an expert in barrier certificate synthesis for dynamical systems. 

Problem Definition:
- Dynamics: {problem.get('dynamics', 'Not specified')}
- Initial Set: {json.dumps(problem.get('initial_set', {}), indent=2)}
- Unsafe Set: {json.dumps(problem.get('unsafe_set', {}), indent=2)}

{previous_templates_info}

Your task is to provide 4 DIVERSE barrier certificate templates that could potentially solve this problem. Be CAREFUL.

Each template should use a DIFFERENT mathematical approach and be DIFFERENT from previous attempts.

**Template Diversity Guidelines:**
- Templates should be tailored to the actual dynamics and geometry
- Use DIFFERENT mathematical structures from previous attempts  
- Consider: DO match template sophistication to system complexity
- Ensure proper algebraic structure with clear coefficient labeling (a1, a2, etc.)
- Be creative with variable combinations and powers

**IMPORTANT:**
- Each template must be MATHEMATICALLY DISTINCT from previous attempts
- All coefficients should be clearly labeled (a1, a2, etc.)
- Provide complete algebraic expressions
- Be creative and mathematically rigorous
- **Match complexity to problem**: SIMPLE dynamics → SIMPLE templates, COMPLEX dynamics → RICHER forms

Format your response as (Don't make it bold):
DYNAMICS_DESCRIPTION: [your analysis]
SET_DESCRIPTION: [your analysis]  
TEMPLATE_1: [exact algebraic template form]
TEMPLATE_1_REASONING: [why this template]
TEMPLATE_2: [exact algebraic template form]
TEMPLATE_2_REASONING: [why this template]
TEMPLATE_3: [exact algebraic template form with advanced features]
TEMPLATE_3_REASONING: [why this template]
TEMPLATE_4: [exact algebraic template form]
TEMPLATE_4_REASONING: [why this template]
MATHEMATICAL_INSIGHTS: [key insights]
"""
        
        print("\n" + "="*80)
        print(f"SENDING ITERATION {iteration} EVOLVED TEMPLATE ANALYSIS PROMPT (x1-x10, 4th Powers):")
        print("="*80)
        print(prompt)
        print("="*80)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text

        print("\n" + "="*80)
        print(f"CLAUDE ITERATION {iteration} EVOLVED TEMPLATE RESPONSE:")
        print("="*80)
        print(content)
        print("="*80)
        
        # Parse the structured response for 4 templates
        analysis = self._parse_multi_template_analysis(content)
        
        logger.info(f"Iteration {iteration} evolved template analysis complete")
        return analysis

    def _process_templates_parallel(self, analysis: ProblemAnalysis, 
                                  problem: Dict[str, Any], iteration: int) -> Dict[int, BarrierSolution]:
        """
        Process all 4 templates in parallel: predict coefficients + refine if needed
        Returns best solution for each template type
        """
        logger.info(f"Processing All 4 Templates Simultaneously (x1-x10 Support)")
        
        # Phase 2a: Predict coefficients for ALL templates in single prompt
        initial_solutions = self._predict_coefficients_all_templates(analysis, problem, iteration)
        
        # Check if any perfect solution found - EARLY STOP HERE
        for template_index, solution in initial_solutions.items():
            if solution and solution.score == 3:
                logger.info(f"Template {template_index + 1} found perfect solution immediately!")
                # Return immediately with just this perfect solution
                return {template_index: solution}
        
        # Phase 2b: Refine each template that needs improvement
        template_solutions = {}
        for template_index in range(4):
            if template_index not in initial_solutions:
                logger.warning(f"Template {template_index + 1} was not generated in initial prediction")
                continue
                
            initial_solution = initial_solutions[template_index]
            best_solution = initial_solution
            
            if initial_solution.score < 3:
                logger.info(f"Refining Template {template_index + 1} solution...")
                template = analysis.get_template(template_index)
                reasoning = analysis.get_reasoning(template_index)
                
                refined_solution = self._refine_single_template_solution(
                    initial_solution, template, reasoning, analysis, problem, iteration
                )
                
                if refined_solution and refined_solution.score > best_solution.score:
                    best_solution = refined_solution
                
                # Check for perfect solution after refinement - EARLY STOP HERE
                if best_solution.score == 3:
                    logger.info(f"Template {template_index + 1} achieved perfect solution after refinement!")
                    return {template_index: best_solution}
            
            # Store best solution for this template
            template_solutions[template_index] = best_solution
            logger.info(f"Template {template_index + 1} best score: {best_solution.score}/3")
        
        logger.info(f"Parallel processing complete. Valid solutions: {len(template_solutions)}/4")
        return template_solutions

    def _predict_coefficients_all_templates(self, analysis: ProblemAnalysis, problem: Dict[str, Any], 
                                          iteration: int) -> Dict[int, BarrierSolution]:
        """
        Predict coefficients for ALL 4 templates in a single prompt - supports x1-x10 and 4th powers
        """
        # Prepare template information
        template_info = []
        for i in range(4):
            template = analysis.get_template(i)
            reasoning = analysis.get_reasoning(i)
            template_info.append(f"""
**Template {i+1}:**
- Mathematical Structure: {template}
- Reasoning: {reasoning}""")
        
        templates_text = "\n".join(template_info)
        
        prompt = f"""You are an expert in barrier certificate synthesis. You need to predict optimal coefficients for ALL 4 barrier certificate templates simultaneously.

Problem Analysis:
- Dynamics: {analysis.dynamics_description}
- Set Geometry: {analysis.set_description}
- Key Insights: {analysis.mathematical_insights}

Original Problem:
- Dynamics: {problem.get('dynamics')}
- Initial Set: {json.dumps(problem.get('initial_set'), indent=2)}
- Unsafe Set: {json.dumps(problem.get('unsafe_set'), indent=2)}

Templates to Process:
{templates_text}


Your task is to predict numerical coefficients for ALL 4 templates that make each one a valid barrier certificate. Each barrier must satisfy:
1. B(x) ≤ 0 for all x in the initial set
2. B(x) > 0 for all x in the unsafe set  
3. ∇B(x)·f(x) < 0 on the boundary where B(x) = 0

For each template, think step by step:
1. Analyze what coefficient values would separate the sets properly
2. Consider the dynamics influence on the gradient condition
3. Ensure the barrier function has the right curvature
5. **For multi-variable systems**: NOTICE contributions from different variables WISELY

Provide complete expressions with numerical coefficients for each template.

Format your response as (Don't make it bold):
TEMPLATE_1_BARRIER: [complete expression with numerical coefficients for template 1]
TEMPLATE_2_BARRIER: [complete expression with numerical coefficients for template 2]  
TEMPLATE_3_BARRIER: [complete expression with numerical coefficients for template 3]
TEMPLATE_4_BARRIER: [complete expression with numerical coefficients for template 4]
"""

        print("\n" + "="*80)
        print(f"SENDING ALL 4 TEMPLATES COEFFICIENT PREDICTION PROMPT (x1-x10, 4th Powers):")
        print("="*80)
        print(prompt)
        print("="*80)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        
        print("\n" + "="*80)
        print(f"CLAUDE ALL 4 TEMPLATES COEFFICIENT PREDICTION RESPONSE:")
        print("="*80)
        print(content)
        print("="*80)
        
        # Extract all 4 barrier certificates from response
        barrier_expressions = self._extract_all_barriers_from_response(content)
        
        # Verify each barrier and create solutions
        template_solutions = {}
        for template_index, barrier_expr in barrier_expressions.items():
            if not barrier_expr:
                logger.warning(f"Failed to extract barrier certificate for template {template_index + 1}")
                continue
            
            # Clean and verify
            barrier_expr = self._final_expression_cleanup(barrier_expr)
            logger.info(f"Template {template_index + 1} predicted: {barrier_expr}")
            
            # SMT Verification
            verification_result = self._verify_barrier_direct(barrier_expr, problem)
            
            if not verification_result:
                logger.warning(f"SMT verification failed for template {template_index + 1}")
                continue
            
            solution = BarrierSolution(
                expression=barrier_expr,
                template_type=analysis.get_template(template_index),
                conditions_satisfied=[
                    verification_result['condition_1'],
                    verification_result['condition_2'], 
                    verification_result['condition_3']
                ],
                verification_details={'conditions': verification_result},
                iteration=iteration,
                phase="initial",
                score=0,  # Will be set in __post_init__
                template_index=template_index
            )
            
            template_solutions[template_index] = solution
            logger.info(f"Template {template_index + 1} verification: {solution.score}/3")
            
            # EARLY STOP: If we found a perfect solution, return immediately
            if solution.score == 3:
                logger.info(f"PERFECT SOLUTION FOUND! Template {template_index + 1} with score 3/3")
                return {template_index: solution}
        
        if not template_solutions:
            raise RuntimeError("All template coefficient predictions failed")
        
        return template_solutions

    def _refine_single_template_solution(self, solution: BarrierSolution, template: str, 
                                       reasoning: str, analysis: ProblemAnalysis, 
                                       problem: Dict[str, Any], iteration: int) -> Optional[BarrierSolution]:
        """
        Refine a single template solution using SMT feedback - supports x1-x10 and 4th powers
        """
        if solution.score == 3:
            return solution  # Already perfect
            
        current_best = solution
        
        for refinement_attempt in range(1, self.max_refinements + 1):
            logger.info(f"Template {solution.template_index + 1} refinement {refinement_attempt}/{self.max_refinements}")
            
            # Analyze failed conditions
            failed_conditions = self._analyze_failed_conditions(current_best)
            
            if not failed_conditions:
                break  # No failed conditions
                
            failed_conditions_str = "\n".join([f"- {fc}" for fc in failed_conditions])
            
            prompt = f"""You are an expert in barrier certificate synthesis. Refine this barrier certificate based on SMT verification feedback.

Current Solution:
- Barrier Certificate: {current_best.expression}
- Template Type: {template}
- Score: {current_best.score}/3 conditions satisfied

Failed Conditions & Counterexamples:
{failed_conditions_str}

Problem:
- Dynamics: {problem.get('dynamics')}
- Initial Set: {json.dumps(problem.get('initial_set'), indent=2)}
- Unsafe Set: {json.dumps(problem.get('unsafe_set'), indent=2)}

Template Reasoning: {reasoning}
Mathematical Insights: {analysis.mathematical_insights}

Based on the counterexamples and failed conditions, your task is to refine the numerical coefficients of barrier certificate to satisfy ALL CONDITIONS:
    - Condition1: B(x) ≤ 0 for all x in the initial set
    - Condition2: B(x) > 0 for all x in the unsafe set  
    - Condition3: ∇B(x)·f(x) < 0 on the boundary where B(x) = 0
    
Maintain the template structure: {template}

Think about:
1. How to adjust coefficients based on the specific counterexamples
2. Which mathematical terms need strengthening or weakening
3. How the dynamics affect the gradient condition
5. **For multi-variable terms**: ENSURE variables contributions WISELY

Format your response as (Don't make it bold):
REFINED_BARRIER_CERTIFICATE: [complete expression with improved coefficients]
"""

            print("\n" + "="*80)
            print(f"SENDING TEMPLATE {solution.template_index + 1} REFINEMENT PROMPT (Attempt {refinement_attempt}):")
            print("="*80)
            print(prompt)
            print("="*80)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            print("\n" + "="*80)
            print(f"CLAUDE TEMPLATE {solution.template_index + 1} REFINEMENT RESPONSE (Attempt {refinement_attempt}):")
            print("="*80)
            print(content)
            print("="*80)
            
            refined_expr = self._extract_barrier_from_response(content)
            
            if not refined_expr:
                logger.warning(f"Failed to extract refined barrier for template {solution.template_index + 1}")
                continue
                
            logger.info(f"Template {solution.template_index + 1} refined: {refined_expr}")
            
            # Verify refined solution
            verification_result = self._verify_barrier_direct(refined_expr, problem)
            
            if verification_result:
                refined_solution = BarrierSolution(
                    expression=refined_expr,
                    template_type=template,
                    conditions_satisfied=[
                        verification_result['condition_1'],
                        verification_result['condition_2'],
                        verification_result['condition_3']
                    ],
                    verification_details={'conditions': verification_result},
                    iteration=iteration,
                    phase=f"refined_{refinement_attempt}",
                    score=0,  # Will be set in __post_init__
                    template_index=solution.template_index
                )
                
                logger.info(f"Template {solution.template_index + 1} refined score: {refined_solution.score}/3")
                
                # Update if improved
                if refined_solution.score > current_best.score:
                    current_best = refined_solution
                    logger.info(f"Template {solution.template_index + 1} improvement! New score: {current_best.score}/3")
                    
                    if current_best.score == 3:
                        logger.info(f"Template {solution.template_index + 1} perfect solution through refinement!")
                        break
            else:
                logger.warning(f"Template {solution.template_index + 1} refined verification failed")
        
        return current_best if current_best.score > solution.score else solution

    def _aggregate_multi_template_solutions(self, template_solutions: Dict[int, BarrierSolution],
                                          analysis: ProblemAnalysis, problem: Dict[str, Any], 
                                          iteration: int) -> Optional[BarrierSolution]:
        """
        Aggregate the best solutions from all 4 templates - supports x1-x10 and 4th powers
        """
        if not template_solutions:
            logger.info("No template solutions available for aggregation")
            return None
            
        # Prepare solution summaries
        solution_summaries = []
        for template_index, solution in template_solutions.items():
            failed_conditions = []
            satisfied_conditions = []
            for j, satisfied in enumerate(solution.conditions_satisfied, 1):
                if not satisfied:
                    failed_conditions.append(f"Condition {j}")
                else:
                    satisfied_conditions.append(f"Condition {j}")
                    
            solution_summaries.append(f"""
Template {template_index + 1} Solution:
- Template Type: {solution.template_type}
- Expression: {solution.expression}
- Score: {solution.score}/3
- Satisfied Conditions: {', '.join(satisfied_conditions) if satisfied_conditions else 'None'}
- Failed Conditions: {', '.join(failed_conditions) if failed_conditions else 'None'}
""")
        
        solutions_text = "\n".join(solution_summaries)
        
        prompt = f"""You are an expert in barrier certificate synthesis. Create a hybrid barrier certificate by intelligently combining the best aspects of these template solutions.

Problem Context:
- Dynamics: {problem.get('dynamics')}
- Initial Set: {json.dumps(problem.get('initial_set'), indent=2)}
- Unsafe Set: {json.dumps(problem.get('unsafe_set'), indent=2)}

Mathematical Insights: {analysis.mathematical_insights}

Template Solutions to Combine:
{solutions_text}

Your task is to create a new barrier certificate that leverages the strengths of each template. Consider:

1. **Identify Strengths**: Which solutions satisfy which conditions?
2. **Smart Combination Strategies**:
   - Weighted combinations: w1*B1(x) + w2*B2(x) + w3*B3(x) + w4*B4(x)
   - Selective term extraction: Take successful terms from each template
   - Mathematical pattern fusion: Combine successful mathematical structures
   - Adaptive combination: Emphasize templates with higher scores
3. **Ensure Validity**: The final expression must satisfy all 3 barrier conditions.

Create a single mathematical expression that intelligently combines the best features.

Format your response as (Don't make it bold):
AGGREGATED_BARRIER_CERTIFICATE: [complete hybrid expression]
"""

        print("\n" + "="*80)
        print(f"SENDING ITERATION {iteration} AGGREGATION PROMPT (x1-x10, 4th Powers):")
        print("="*80)
        print(prompt)
        print("="*80)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.content[0].text
        
        print("\n" + "="*80)
        print(f"CLAUDE ITERATION {iteration} AGGREGATION RESPONSE:")
        print("="*80)
        print(content)
        print("="*80)
        
        aggregated_expr = self._extract_barrier_from_response(content)
        
        if not aggregated_expr:
            raise ValueError(f"Failed to extract aggregated barrier certificate from LLM response")
            
        logger.info(f"Aggregated barrier: {aggregated_expr}")
        
        # Verify aggregated solution
        verification_result = self._verify_barrier_direct(aggregated_expr, problem)
        
        if not verification_result:
            logger.warning(f"Aggregated solution verification failed")
            return None
        
        aggregated_solution = BarrierSolution(
            expression=aggregated_expr,
            template_type="hybrid_aggregated",
            conditions_satisfied=[
                verification_result['condition_1'],
                verification_result['condition_2'],
                verification_result['condition_3']
            ],
            verification_details={'conditions': verification_result},
            iteration=iteration,
            phase="aggregated",
            score=0,  # Will be set in __post_init__
            template_index=-1  # Special index for aggregated solutions
        )
        
        logger.info(f"Aggregated solution score: {aggregated_solution.score}/3")
        return aggregated_solution

    def _parse_multi_template_analysis(self, content: str) -> ProblemAnalysis:
        """Parse structured analysis response with 4 templates - supports x1-x10"""
        lines = content.split('\n')
        analysis_data = {}
        
        # Enhanced parsing for 4 templates
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            
            # Check for section headers
            if line.startswith('DYNAMICS_DESCRIPTION:'):
                if current_key:
                    analysis_data[current_key] = ' '.join(current_value)
                current_key = 'dynamics_description'
                current_value = [line.split(':', 1)[1].strip()] if ':' in line else []
                
            elif line.startswith('SET_DESCRIPTION:'):
                if current_key:
                    analysis_data[current_key] = ' '.join(current_value)
                current_key = 'set_description'
                current_value = [line.split(':', 1)[1].strip()] if ':' in line else []
                
            elif line.startswith('TEMPLATE_1:'):
                if current_key:
                    analysis_data[current_key] = ' '.join(current_value)
                current_key = 'template_1'
                current_value = [line.split(':', 1)[1].strip()] if ':' in line else []
                
            elif line.startswith('TEMPLATE_1_REASONING:'):
                if current_key:
                    analysis_data[current_key] = ' '.join(current_value)
                current_key = 'template_1_reasoning'
                current_value = [line.split(':', 1)[1].strip()] if ':' in line else []
                
            elif line.startswith('TEMPLATE_2:'):
                if current_key:
                    analysis_data[current_key] = ' '.join(current_value)
                current_key = 'template_2'
                current_value = [line.split(':', 1)[1].strip()] if ':' in line else []
                
            elif line.startswith('TEMPLATE_2_REASONING:'):
                if current_key:
                    analysis_data[current_key] = ' '.join(current_value)
                current_key = 'template_2_reasoning'
                current_value = [line.split(':', 1)[1].strip()] if ':' in line else []
                
            elif line.startswith('TEMPLATE_3:'):
                if current_key:
                    analysis_data[current_key] = ' '.join(current_value)
                current_key = 'template_3'
                current_value = [line.split(':', 1)[1].strip()] if ':' in line else []
                
            elif line.startswith('TEMPLATE_3_REASONING:'):
                if current_key:
                    analysis_data[current_key] = ' '.join(current_value)
                current_key = 'template_3_reasoning'
                current_value = [line.split(':', 1)[1].strip()] if ':' in line else []
                
            elif line.startswith('TEMPLATE_4:'):
                if current_key:
                    analysis_data[current_key] = ' '.join(current_value)
                current_key = 'template_4'
                current_value = [line.split(':', 1)[1].strip()] if ':' in line else []
                
            elif line.startswith('TEMPLATE_4_REASONING:'):
                if current_key:
                    analysis_data[current_key] = ' '.join(current_value)
                current_key = 'template_4_reasoning'
                current_value = [line.split(':', 1)[1].strip()] if ':' in line else []
                
            elif line.startswith('MATHEMATICAL_INSIGHTS:'):
                if current_key:
                    analysis_data[current_key] = ' '.join(current_value)
                current_key = 'mathematical_insights'
                current_value = [line.split(':', 1)[1].strip()] if ':' in line else []
                
            elif current_key and line:  # Continue current section
                current_value.append(line)
        
        # Add the last section
        if current_key:
            analysis_data[current_key] = ' '.join(current_value)
        
        # Extract templates and reasoning
        templates = []
        reasoning = []
        
        for i in range(1, 5):
            template_key = f'template_{i}'
            reasoning_key = f'template_{i}_reasoning'
            
            template = analysis_data.get(template_key)
            template_reasoning = analysis_data.get(reasoning_key)
            
            if not template:
                raise ValueError(f"Template {i} not found in LLM response")
            if not template_reasoning:
                raise ValueError(f"Template {i} reasoning not found in LLM response")
            
            # Clean template - enhanced for x1-x10 support
            template = self._clean_extracted_template(template)
            
            templates.append(template)
            reasoning.append(template_reasoning)
        
        dynamics_desc = analysis_data.get('dynamics_description')
        set_desc = analysis_data.get('set_description')
        math_insights = analysis_data.get('mathematical_insights')
        
        if not dynamics_desc:
            raise ValueError("Dynamics description not found in LLM response")
        if not set_desc:
            raise ValueError("Set description not found in LLM response")
        if not math_insights:
            raise ValueError("Mathematical insights not found in LLM response")
        
        return ProblemAnalysis(
            dynamics_description=dynamics_desc,
            set_description=set_desc,
            suggested_templates=templates,
            template_reasoning=reasoning,
            mathematical_insights=math_insights
        )

    def _clean_extracted_template(self, template: str) -> str:
        """Clean extracted template expression - enhanced for x1-x10 and 4th powers"""
        if not template:
            raise ValueError("Empty template provided for cleaning")
        
        # Basic cleaning
        template = template.strip().strip('"\'.,;:')
        
        # Convert Unicode if present - enhanced for x1-x10
        unicode_map = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5',
            '₆': '6', '₇': '7', '₈': '8', '₉': '9',
            '²': '**2', '³': '**3', '⁴': '**4', '⁵': '**5', '⁶': '**6'
        }
        for unicode_char, replacement in unicode_map.items():
            template = template.replace(unicode_char, replacement)
        
        # Convert ^ to ** for all powers including 4th
        template = re.sub(r'\^(\d+)', r'**\1', template)
        template = template.replace('^', '**')
        
        return template

    def _verify_barrier_direct(self, barrier_expr: str, problem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Enhanced verification: sample-based pre-check followed by SMT if needed
        """
        print(f"DEBUG: Enhanced barrier verification starting...")
        print(f"DEBUG: Barrier expression: {barrier_expr}")

        try:
            # Step 1: Generate samples for validation
            print(f"DEBUG: Step 1 - Generating samples...")
            samples = self._generate_samples_for_barrier_validation(problem, num_samples=1000)
            
            # Step 2: Sample-based pre-validation
            print(f"DEBUG: Step 2 - Sample-based pre-validation...")
            sample_validation = self._validate_barrier_on_samples(barrier_expr, problem, samples)
            
            if not sample_validation['success']:
                print(f"DEBUG: Sample validation failed with error: {sample_validation.get('error', 'Unknown')}")
                return None
            
            sample_score = sample_validation['score']
            print(f"DEBUG: Sample validation score: {sample_score}/3")
            
            # Step 3: Decision logic
            if sample_score < 3:
                print(f"DEBUG: Sample validation failed ({sample_score}/3) - skipping SMT verification")
                return {
                    'condition_1': sample_validation['conditions_satisfied'][0],
                    'condition_2': sample_validation['conditions_satisfied'][1],
                    'condition_3': sample_validation['conditions_satisfied'][2],
                    'validation_method': 'sample_only',
                    'sample_score': sample_score,
                    'verification_details': {
                        'sample_based': sample_validation,
                        'smt_based': None
                    }
                }
            
            # Step 4: Proceed to SMT verification since sample validation passed
            print(f"DEBUG: Sample validation passed (3/3) - proceeding to SMT verification...")
            smt_validation = get_detailed_condition_results(
                barrier_expr,
                problem['initial_set'],
                problem['unsafe_set'], 
                problem['dynamics']
            )
            
            if smt_validation['success']:
                print(f"DEBUG: SMT verification completed successfully")
                return {
                    'condition_1': smt_validation['conditions']['condition_1'],
                    'condition_2': smt_validation['conditions']['condition_2'],
                    'condition_3': smt_validation['conditions']['condition_3'],
                    'validation_method': 'sample_then_smt',
                    'sample_score': sample_score,
                    'verification_details': {
                        'sample_based': sample_validation,
                        'smt_based': smt_validation
                    }
                }
            else:
                print(f"DEBUG: SMT verification failed despite sample validation passing")
                return None
                
        except Exception as e:
            print(f"ERROR: Enhanced verification failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_failed_conditions(self, solution: BarrierSolution) -> List[str]:
        """Analyze which conditions failed and extract counterexamples from both sample-based and SMT validation"""
        failed_conditions = []
        verification_details = solution.verification_details.get('conditions', {})

        # Extract sample-based counterexamples if available
        sample_based_details = verification_details.get('verification_details', {}).get('sample_based', {})
        sample_counterexamples = sample_based_details.get('counterexamples', {})

        # Extract SMT-based counterexamples if available  
        smt_based_details = verification_details.get('verification_details', {}).get('smt_based', {})

        for i, (condition_key, satisfied) in enumerate([
            ('condition_1', solution.conditions_satisfied[0]),
            ('condition_2', solution.conditions_satisfied[1]), 
            ('condition_3', solution.conditions_satisfied[2])
        ], 1):
            if not satisfied:
                condition_analysis = []
                
                # Add SMT counterexample if available
                if smt_based_details and 'conditions' in smt_based_details:
                    smt_condition_details = smt_based_details['conditions'].get(condition_key, {})
                    smt_counterexample = smt_condition_details.get('counterexample', 'None available')
                    if smt_counterexample != 'None available':
                        condition_analysis.append(f"Counterexample: {smt_counterexample}")
                
                # Add sample-based counterexamples (limit to 3)
                sample_condition_key = f'condition_{i}'
                if sample_condition_key in sample_counterexamples:
                    sample_violations = sample_counterexamples[sample_condition_key][:3]  # Limit to 3
                    if sample_violations:
                        for j, violation in enumerate(sample_violations, 1):
                            if condition_key == 'condition_1':
                                # Condition 1: B(x) <= 0 for x in X0
                                point = violation.get('point', [])
                                barrier_value = violation.get('barrier_value', 'N/A')
                                violation_amount = violation.get('violation', 'N/A')
                                condition_analysis.append(f"Point x={point}")
                                
                            elif condition_key == 'condition_2':
                                # Condition 2: B(x) > 0 for x in Xu  
                                point = violation.get('point', [])
                                barrier_value = violation.get('barrier_value', 'N/A')
                                violation_amount = violation.get('violation', 'N/A')
                                condition_analysis.append(f"Point x={point}")
                                
                            elif condition_key == 'condition_3':
                                # Condition 3: ∇B(x)·f(x) < 0 (dynamics condition)
                                trajectory = violation.get('trajectory', [])
                                barrier_x = violation.get('barrier_x', 'N/A')
                                barrier_next = violation.get('barrier_next', 'N/A')
                                violation_amount = violation.get('violation', 'N/A')
                                if len(trajectory) >= 2:
                                    x_point = trajectory[0]
                                    condition_analysis.append(f"Point x={x_point}")
                
                # Combine all analysis for this condition
                if condition_analysis:
                    failed_condition_text = f"Condition {i} violations: " + " | ".join(condition_analysis)
                    failed_conditions.append(failed_condition_text)
                else:
                    # Fallback if no detailed counterexamples available
                    condition_details = verification_details.get(condition_key, {})
                    failed_conditions.append(f"Condition {i}: {condition_details.get('details', 'Failed')} | No detailed counterexamples available")

        return failed_conditions

    def _extract_barrier_from_response(self, content: str) -> Optional[str]:
        """Extract barrier certificate from LLM response - enhanced for x1-x10"""
        # Try structured patterns first
        structured_patterns = [
            r'BARRIER_CERTIFICATE\s*:\s*(.+?)(?:\n|$)',
            r'REFINED_BARRIER_CERTIFICATE\s*:\s*(.+?)(?:\n|$)',
            r'AGGREGATED_BARRIER_CERTIFICATE\s*:\s*(.+?)(?:\n|$)',
            r'TEMPLATE_\d+_BARRIER\s*:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in structured_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            if matches:
                for match in matches:
                    expression = match.strip()
                    cleaned_expr = self._clean_extracted_expression(expression)
                    if cleaned_expr and self._validate_extracted_expression(cleaned_expr, strict=True):
                        return cleaned_expr
        
        # If structured patterns fail, look for complete mathematical expressions
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            # Look for lines that look like complete barrier expressions - enhanced for x1-x10
            if (len(line) > 10 and 
                re.search(r'x\d+', line) and  # Any variable x1-x10
                any(op in line for op in ['+', '-', '*']) and
                re.search(r'\d', line) and
                not any(word in line.lower() for word in ['template', 'barrier', 'certificate', 'expression', 'analysis'])):
                
                cleaned_expr = self._clean_extracted_expression(line)
                if cleaned_expr and self._validate_extracted_expression(cleaned_expr, strict=True):
                    return cleaned_expr
        
        return None

    def _clean_extracted_expression(self, expression: str) -> str:
        """Clean extracted expression - enhanced for x1-x10 and 4th powers"""
        if not expression:
            return ""
        
        # Remove backticks and formatting
        expression = expression.replace('`', '').replace('**', '**')
        
        # Remove common prefixes but keep the math
        expression = re.sub(r'^.*?BARRIER.*?:\s*', '', expression, flags=re.IGNORECASE)
        expression = re.sub(r'^.*?B\(.*?\)\s*=\s*', '', expression, flags=re.IGNORECASE)
        
        # Convert Unicode - enhanced for x1-x10
        unicode_map = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5',
            '₆': '6', '₇': '7', '₈': '8', '₉': '9',
            '²': '**2', '³': '**3', '⁴': '**4', '⁵': '**5', '⁶': '**6'
        }
        for unicode_char, replacement in unicode_map.items():
            expression = expression.replace(unicode_char, replacement)
        
        # Convert ^ to ** for all powers
        expression = re.sub(r'\^(\d+)', r'**\1', expression)
        expression = expression.replace('^', '**')
        
        # Clean up parentheses around single numbers: (-0.5) -> -0.5
        expression = re.sub(r'\((-?\d+\.?\d*)\)', r'\1', expression)
        
        # Fix missing multiplication signs - enhanced for x1-x10
        expression = re.sub(r'(\d)([x])', r'\1*\2', expression)
        
        # Clean up multiple spaces
        expression = re.sub(r'\s+', ' ', expression).strip()
        
        # Remove trailing punctuation
        expression = expression.rstrip('.,;:')
        
        # Ensure the expression doesn't start with operators (except unary minus)
        if re.match(r'^\s*[+*/]', expression):
            return ""
        
        return expression

    def _validate_extracted_expression(self, expression: str, strict: bool = False) -> bool:
        """Validate extracted expression - enhanced for x1-x10"""
        if not expression or len(expression) < 3:
            return False
        
        # Must contain variables x1-x10
        has_variables = bool(re.search(r'x\d+', expression))
        if not has_variables:
            return False
        
        # Must have numerical coefficients
        if not re.search(r'\d', expression):
            return False
        
        return True

    def _final_expression_cleanup(self, expression: str) -> str:
        """Final expression cleanup - enhanced for x1-x10 and 4th powers"""
        if not expression:
            return ""
        
        # Basic cleanup
        expression = expression.strip()
        
        # Fix spacing around operators
        expression = re.sub(r'\s*\+\s*', ' + ', expression)
        expression = re.sub(r'\s*-\s*', ' - ', expression)
        expression = re.sub(r'\s*\*\s*', '*', expression)
        
        # Ensure we don't have double operators
        expression = re.sub(r'\+\s*\+', '+', expression)
        expression = re.sub(r'-\s*\+', '-', expression)
        expression = re.sub(r'\+\s*-', '+ -', expression)
        
        # Clean up the expression but preserve structure
        expression = expression.strip()
        
        return expression

    def _extract_all_barriers_from_response(self, content: str) -> Dict[int, str]:
        """Extract all 4 barrier certificates from a single LLM response - enhanced for x1-x10"""
        barrier_expressions = {}
        
        # Patterns for extracting each template's barrier
        template_patterns = [
            r'TEMPLATE_1_BARRIER\s*:\s*(.+?)(?:\n(?!.*\d)|$)',
            r'TEMPLATE_2_BARRIER\s*:\s*(.+?)(?:\n(?!.*\d)|$)',
            r'TEMPLATE_3_BARRIER\s*:\s*(.+?)(?:\n(?!.*\d)|$)',
            r'TEMPLATE_4_BARRIER\s*:\s*(.+?)(?:\n(?!.*\d)|$)'
        ]
        
        for template_index, pattern in enumerate(template_patterns):
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if matches:
                for match in matches:
                    # Take the full match, not just first line
                    expression = match.strip()
                    
                    # Remove any continuation text that might be captured
                    for stop_phrase in ['\n\n', 'Template', 'For template', 'This barrier']:
                        if stop_phrase in expression:
                            expression = expression.split(stop_phrase)[0].strip()
                    
                    cleaned_expr = self._clean_extracted_expression(expression)
                    if cleaned_expr and self._validate_extracted_expression(cleaned_expr, strict=True):
                        barrier_expressions[template_index] = cleaned_expr
                        logger.info(f"Extracted Template {template_index + 1}: {cleaned_expr}")
                        break
                else:
                    logger.warning(f"Failed to extract valid expression for Template {template_index + 1}")
            else:
                logger.warning(f"No pattern match found for Template {template_index + 1}")
        
        if not barrier_expressions:
            raise ValueError("Failed to extract any barrier certificates from LLM response")
        
        return barrier_expressions

    def _update_best_solution(self, solution: BarrierSolution):
        """Update the global best solution"""
        if self.best_solution is None or solution.score > self.best_solution.score:
            self.best_solution = solution
            logger.info(f"New best solution! Score: {solution.score}/3, Template: {solution.template_index + 1}")

    def _solution_to_dict(self, solution: Optional[BarrierSolution]) -> Optional[Dict[str, Any]]:
        """Convert BarrierSolution to dictionary"""
        if solution is None:
            return None
            
        return {
            'expression': solution.expression,
            'template_type': solution.template_type,
            'template_index': solution.template_index,
            'conditions_satisfied': solution.conditions_satisfied,
            'score': solution.score,
            'iteration': solution.iteration,
            'phase': solution.phase,
            'verification_details': solution.verification_details
        }

    def _reset_state(self):
        """Reset internal state for new problem"""
        self.current_iteration_solutions.clear()
        self.all_template_history.clear()
        self.analysis_history.clear()
        self.best_solution = None


# Example usage
if __name__ == "__main__":
    test_problem = {'dynamics':'dx1/dt = -x1, dx2/dt = -x2, dx3/dt = -x3',
        'initial_set':{
            'type': 'ball',
            'radius': 1.0,
            'center': [0, 0, 0]
        },
        'unsafe_set':{
            'type': 'ball',
            'radius': 3.0,
            'center': [0, 0, 0],
            'complement': True
        },
        'description':"3D linear stability system - Fossil 2.0 benchmark"}

    # API configuration
    api_key = "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"
    
    # Initialize enhanced framework
    enhanced_synthesizer = EnhancedGoTBarrierSynthesis(
        api_key=api_key,
        model="claude-sonnet-4-20250514",
        max_iterations=1,
        max_refinements=1
    )
    
    # Run synthesis
    print("Starting Enhanced GoT Barrier Certificate Synthesis (x1-x10, 4th Powers Support)...")
    result = enhanced_synthesizer.synthesize_barrier_certificate(test_problem)
    
    # Display results
    print("\n" + "="*80)
    print("ENHANCED SYNTHESIS RESULTS (x1-x10, 4th Powers)")
    print("="*80)
    
    if result['success']:
        print(f"SUCCESS!")
        print(f"Barrier Certificate: {result['barrier_certificate']}")
        print(f"Template Type: {result['template_type']}")
        print(f"Found in Iteration: {result['iteration_found']}")
        print(f"Total Time: {result['total_time']:.2f} seconds")
        print(f"Templates Explored: {result['templates_explored']}")
    else:
        print(f"FAILED to find valid barrier certificate")
        if result.get('best_solution'):
            print(f"Best Score: {result['best_solution']['score']}/3")
            print(f"Best Template: {result['best_solution'].get('template_index', 'Unknown') + 1}")
        print(f"Total Time: {result['total_time']:.2f} seconds")
        print(f"Templates Explored: {result['templates_explored']}")
    
    print(f"Total Iterations: {result['total_iterations']}")
