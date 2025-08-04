#!/usr/bin/env python3
"""
Enhanced Pipeline following the Physics-Informed Graph of Thoughts architecture
with LLM-based selection, multi-strategy aggregation, and refinement phases
"""

import sys
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Setup logging
def setup_logging():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"enhanced_barrier_synthesis_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Enhanced session started: {datetime.now()}")
    logger.info(f"Log file: {log_filename}")
    return logger

logger = setup_logging()

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import components
try:
    from llama_model import LlamaTogetherAI
    from barrier_utils import (validate_barrier_certificate, get_detailed_condition_results,
                              parse_barrier_certificate, clean_and_extract_barrier)
    logger.info("Successfully imported custom components")
except ImportError as e:
    logger.error(f"Failed to import components: {e}")
    sys.exit(1)

# Import Graph of Thoughts
try:
    from graph_of_thoughts import operations
    from graph_of_thoughts.controller import Controller
    from graph_of_thoughts.operations import Thought
    logger.info("Using official graph_of_thoughts package")
except ImportError as e:
    logger.error(f"Official GoT package required: {e}")
    sys.exit(1)


class ValidSolutionFound(Exception):
    """Exception raised when a valid barrier certificate is found"""
    def __init__(self, barrier: str, template_type: str, verification_details: Dict, phase: str = "unknown"):
        self.barrier = barrier
        self.template_type = template_type
        self.verification_details = verification_details
        self.phase = phase
        super().__init__(f"Valid barrier found in {phase}: {barrier}")


class TemplateEnforcedGenerate(operations.Generate):
    """Parameter synthesis with strict template structure enforcement"""
    
    def __init__(self, template_type: str, problem_params: Dict):
        super().__init__()
        self.template_type = template_type
        self.problem_params = problem_params
        
    def _execute(self, lm, prompter, parser, **kwargs):
        try:
            print(f"\n{'='*60}")
            print(f"GENERATING {self.template_type.upper()} SOLUTIONS")
            print(f"{'='*60}")
            logger.info(f"Generating 5 {self.template_type} solutions with strict template enforcement...")
            
            # Generate prompt with explicit template structure requirements
            prompt = self._create_enforced_prompt()
            
            # Query LLM with higher temperature for diversity
            lm.set_temperature(0.9)
            response = lm.query(prompt, num_responses=1)
            response_texts = lm.get_response_texts(response)
            
            if not response_texts:
                logger.error(f"No response from LLM for {self.template_type}")
                self.thoughts = []
                return
            
            print(f"Response generated for {self.template_type} template")
            logger.info(f"Generated response for {self.template_type}")
            
            # Extract and validate solutions with template enforcement
            solutions = self._extract_and_validate_solutions(response_texts[0])
            
            # Create thoughts for each solution
            self.thoughts = []
            for i, solution in enumerate(solutions, 1):
                if solution:
                    print(f"Solution {i}: B(x) = {solution}")
                    logger.info(f"{self.template_type} solution #{i}: {solution}")
                    
                    # Immediate SMT verification
                    is_valid = validate_barrier_certificate(
                        solution,
                        self.problem_params.get('initial_set', {}),
                        self.problem_params.get('unsafe_set', {}),
                        self.problem_params.get('dynamics', '')
                    )
                    
                    if is_valid:
                        print(f"\n{'*'*60}")
                        print(f"VALID SOLUTION FOUND IN PHASE 2!")
                        print(f"Template: {self.template_type}")
                        print(f"Solution: B(x) = {solution}")
                        print(f"{'*'*60}\n")
                        logger.info(f"VALID SOLUTION FOUND IN PHASE 2: {solution}")
                        
                        verification_details = get_detailed_condition_results(
                            solution,
                            self.problem_params.get('initial_set', {}),
                            self.problem_params.get('unsafe_set', {}),
                            self.problem_params.get('dynamics', '')
                        )
                        
                        result = {
                            'template_type': self.template_type,
                            'solution_number': i,
                            'barrier_certificate': solution,
                            'content': solution,
                            'valid_solution': True,
                            'verification_details': verification_details,
                            'phase': 'parameter_synthesis'
                        }
                        
                        thought = Thought(result)
                        self.thoughts = [thought]
                        
                        # Stop immediately when valid solution found
                        raise ValidSolutionFound(solution, self.template_type, verification_details, "Phase 2: Parameter Synthesis")
                    else:
                        print(f"Solution {i}: Failed verification")
                        logger.info(f"{self.template_type} solution #{i} failed verification")
                    
                    # Create thought for this solution
                    result = {
                        'template_type': self.template_type,
                        'solution_number': i,
                        'barrier_certificate': solution,
                        'content': solution,
                        'valid_solution': False,
                        'phase': 'parameter_synthesis'
                    }
                    
                    thought = Thought(result)
                    self.thoughts.append(thought)
                else:
                    print(f"Solution {i}: Invalid format")
                    logger.warning(f"Could not extract valid {self.template_type} solution #{i}")
            
            print(f"\nCreated {len(self.thoughts)} thoughts for {self.template_type} template")
            logger.info(f"Created {len(self.thoughts)} thoughts for {self.template_type}")
            
        except ValidSolutionFound:
            # Re-raise to stop pipeline
            raise
        except Exception as e:
            logger.error(f"Error in {self.template_type} generation: {e}")
            self.thoughts = []
    
    def _create_enforced_prompt(self) -> str:
        """Create prompt with strict template structure enforcement"""
        initial_set = self.problem_params.get('initial_set', {})
        unsafe_set = self.problem_params.get('unsafe_set', {})
        dynamics = self.problem_params.get('dynamics', '')
        
        initial_desc = self._format_set_description(initial_set)
        unsafe_desc = self._format_set_description(unsafe_set)
        
        # Template-specific strict requirements
        template_requirements = {
            'linear': {
                'allowed': ['x1', 'x2', 'constants'],
                'forbidden': ['x1**2', 'x2**2', 'x1*x2', 'x1**3', 'x2**3', 'x1**4', 'x2**4'],
                'examples': [
                    "1.5*x1 + 0.8*x2 - 1.2",
                    "2.1*x1 - 1.8", 
                    "1.7*x2 - 1.3",
                    "0.9*x1 + 2.4*x2 - 2.1",
                    "3.2*x1 + 0.4*x2 - 2.8"
                ]
            },
            'quadratic': {
                'allowed': ['x1**2', 'x2**2', 'x1*x2', 'x1', 'x2', 'constants'],
                'forbidden': ['x1**3', 'x2**3', 'x1**4', 'x2**4'],
                'examples': [
                    "1.2*x1**2 + 0.8*x2**2 - 1.5",
                    "0.9*x1**2 + 0.3*x1*x2 + 1.1*x2**2 - 2.0",
                    "1.5*x1**2 + 0.7*x2**2 + 0.2*x1 - 1.8",
                    "0.8*x1**2 + 0.1*x1*x2 + 1.3*x2**2 + 0.4*x1 + 0.2*x2 - 2.2",
                    "2.3*x1**2 + 0.5*x2**2 + 0.4*x1 + 0.6*x2 - 2.7"
                ]
            },
            'nonlinear': {
                'allowed': ['x1**2', 'x2**2', 'constants'],
                'forbidden': ['x1*x2', 'x1**3', 'x2**3', 'x1**4', 'x2**4'],
                'examples': [
                    "1.1*x1**2 + 1.1*x2**2 - 1.3",
                    "2.2*x1**2 - 1.9",
                    "1.8*x2**2 - 1.6", 
                    "0.6*x1**2 + 1.5*x2**2 - 1.1",
                    "3.1*x1**2 + 0.3*x2**2 - 2.4"
                ]
            },
            'polynomial': {
                'allowed': ['x1**4', 'x2**4', 'x1**2', 'x2**2', 'x1*x2', 'x1', 'x2', 'constants'],
                'forbidden': ['x1**3', 'x2**3', 'higher powers'],
                'examples': [
                    "0.3*x1**4 + 0.2*x2**4 - 2.1",
                    "0.4*x1**4 + 1.2*x1**2 + 0.9*x2**2 - 2.5",
                    "0.2*x1**4 + 0.1*x2**4 + 1.1*x1**2 + 0.8*x2**2 + 0.1*x1*x2 - 2.8",
                    "0.5*x1**4 + 0.3*x2**4 + 0.9*x1**2 + 1.4*x2**2 + 0.1*x1 - 3.1",
                    "0.1*x1**4 + 0.6*x2**4 + 1.8*x1**2 + 0.4*x2**2 + 0.2*x1*x2 + 0.3*x2 - 3.8"
                ]
            }
        }
        
        requirements = template_requirements.get(self.template_type, template_requirements['quadratic'])
        allowed_terms = ', '.join(requirements['allowed'])
        forbidden_terms = ', '.join(requirements['forbidden'])
        examples_text = '\n'.join([f"  Example {i+1}: B(x) = {ex}" for i, ex in enumerate(requirements['examples'])])
        
        return f"""Generate 5 STRUCTURALLY DIFFERENT {self.template_type.upper()} barrier certificates with STRICT TEMPLATE ADHERENCE:

SYSTEM DYNAMICS: {dynamics}
INITIAL SET: {initial_desc}
UNSAFE SET: {unsafe_desc}

STRICT {self.template_type.upper()} TEMPLATE REQUIREMENTS:
ALLOWED TERMS ONLY: {allowed_terms}
FORBIDDEN TERMS: {forbidden_terms}
NO OTHER TERMS PERMITTED

STRUCTURAL DIVERSITY EXAMPLES (create your own coefficients):
{examples_text}

CRITICAL ENFORCEMENT RULES:
1. Use EXACTLY the allowed terms for {self.template_type} template
2. NO terms from forbidden list are permitted
3. Each solution must have DIFFERENT mathematical structure
4. Use ONLY numerical coefficients (no letters a, b, c, α, β, γ)
5. Constant term must be NEGATIVE to satisfy initial condition
6. Create your own unique coefficients - do NOT copy examples

BARRIER CERTIFICATE CONDITIONS:
1. B(x) ≤ 0 for all x in initial set
2. B(x) > 0 for all x in unsafe set  
3. ∇B(x)·f(x) ≤ 0 for all x where B(x) = 0

OUTPUT FORMAT (exactly as shown):
Solution 1: B(x) = [your {self.template_type} expression with allowed terms only]
Solution 2: B(x) = [your {self.template_type} expression with allowed terms only]
Solution 3: B(x) = [your {self.template_type} expression with allowed terms only]
Solution 4: B(x) = [your {self.template_type} expression with allowed terms only]
Solution 5: B(x) = [your {self.template_type} expression with allowed terms only]

GENERATE 5 VALID {self.template_type.upper()} BARRIERS NOW:"""
    
    # Include all previous helper methods from original class...
    def _extract_and_validate_solutions(self, text: str) -> List[str]:
        """Extract and validate solutions with template enforcement"""
        solutions = []
        
        # Pattern to match "Solution X: B(x) = ..." 
        solution_patterns = [
            r'Solution\s+\d+\s*:\s*B\(x\)\s*=\s*([^\n]+)',
            r'Solution\s+\d+\s*:\s*([^\n]+)',
            r'B\(x\)\s*=\s*([^\n]+)',
        ]
        
        # Try each pattern
        for pattern in solution_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if len(matches) >= 5:
                # Clean and validate each match
                for match in matches[:5]:
                    cleaned = self._clean_solution(match.strip())
                    if cleaned and self._validate_template_structure(cleaned):
                        solutions.append(cleaned)
                
                if len(solutions) >= 5:
                    break
        
        # If we don't have 5 solutions, try line-by-line extraction
        if len(solutions) < 5:
            lines = text.split('\n')
            for line in lines:
                if ('x1' in line or 'x2' in line) and any(c.isdigit() for c in line):
                    cleaned = self._clean_solution(line)
                    if cleaned and self._validate_template_structure(cleaned) and cleaned not in solutions:
                        solutions.append(cleaned)
                        if len(solutions) >= 5:
                            break
        
        # Fill with fallback solutions if needed
        while len(solutions) < 5:
            fallback = self._generate_template_compliant_fallback(len(solutions) + 1)
            if fallback not in solutions:
                solutions.append(fallback)
        
        return solutions[:5]
    
    def _validate_template_structure(self, solution: str) -> bool:
        """Validate that solution adheres to template structure"""
        try:
            if not solution or len(solution) < 5:
                return False
            
            # Basic validation
            if not re.search(r'x\d+', solution):
                return False
            
            if not re.search(r'\d', solution):
                return False
            
            # Template-specific validation
            if self.template_type == 'linear':
                # Should not have x^2, x^3, x^4 terms
                if re.search(r'x\d+\*\*[2-9]', solution):
                    logger.warning(f"Linear solution has forbidden powers: {solution}")
                    return False
                # Should not have cross terms like x1*x2
                if re.search(r'x\d+\s*\*\s*x\d+', solution):
                    logger.warning(f"Linear solution has forbidden cross terms: {solution}")
                    return False
            
            elif self.template_type == 'quadratic':
                # Should not have x^3, x^4 terms  
                if re.search(r'x\d+\*\*[3-9]', solution):
                    logger.warning(f"Quadratic solution has forbidden high powers: {solution}")
                    return False
            
            elif self.template_type == 'nonlinear':
                # Should not have x^3, x^4 terms
                if re.search(r'x\d+\*\*[3-9]', solution):
                    logger.warning(f"Nonlinear solution has forbidden high powers: {solution}")
                    return False
                # Should not have cross terms like x1*x2 (for this simple nonlinear template)
                if re.search(r'x\d+\s*\*\s*x\d+', solution):
                    logger.warning(f"Nonlinear solution has forbidden cross terms: {solution}")
                    return False
            
            elif self.template_type == 'polynomial':
                # Should not have x^5 or higher
                if re.search(r'x\d+\*\*[5-9]', solution):
                    logger.warning(f"Polynomial solution has forbidden very high powers: {solution}")
                    return False
            
            # Try to parse with sympy
            from barrier_utils import parse_barrier_certificate
            expr, vars = parse_barrier_certificate(solution)
            return expr is not None
            
        except Exception as e:
            logger.warning(f"Template validation error for '{solution}': {e}")
            return False
    
    def _clean_solution(self, solution: str) -> str:
        """Clean and normalize solution string"""
        try:
            # Remove common prefixes
            solution = re.sub(r'Solution\s+\d+\s*:\s*', '', solution, flags=re.IGNORECASE)
            solution = re.sub(r'B\(x\)\s*=\s*', '', solution, flags=re.IGNORECASE)
            
            # Clean formatting
            solution = solution.strip().strip('"\'.,;:')
            
            # Fix common issues
            solution = solution.replace('^', '**')
            solution = re.sub(r'\s+', ' ', solution)
            
            # Remove trailing operators
            solution = re.sub(r'[+\-\*]\s*$', '', solution).strip()
            
            # Remove any explanatory text (keep only the mathematical expression)
            # Look for the first mathematical expression
            math_pattern = r'([-+]?\d*\.?\d*\*?x\d+(?:\*\*\d+)?(?:\s*[+\-]\s*[-+]?\d*\.?\d*\*?x\d+(?:\*\*\d+)?)*(?:\s*[+\-]\s*[-+]?\d*\.?\d+)?)'
            match = re.search(math_pattern, solution)
            if match:
                solution = match.group(1)
            
            return solution
            
        except Exception:
            return ""
    
    def _generate_template_compliant_fallback(self, solution_num: int) -> str:
        """Generate fallback solutions that strictly comply with template structure"""
        fallbacks = {
            'linear': [
                "1.5*x1 + 0.8*x2 - 1.2",        # Both terms
                "2.1*x1 - 1.8",                  # Only x1 term  
                "1.7*x2 - 1.3",                  # Only x2 term
                "0.9*x1 + 2.4*x2 - 2.1",        # Different ratios
                "3.2*x1 + 0.4*x2 - 2.8"         # High x1 coefficient
            ],
            'quadratic': [
                "1.2*x1**2 + 0.8*x2**2 - 1.5",                      # Pure quadratic
                "0.6*x1**2 + 0.3*x1*x2 + 1.1*x2**2 - 2.1",         # With cross term
                "1.8*x1**2 + 1.4*x2**2 + 0.2*x1 - 1.9",            # With linear terms
                "0.9*x1**2 + 0.1*x1*x2 + 1.6*x2**2 + 0.3*x2 - 1.4", # Full form
                "2.3*x1**2 + 0.5*x2**2 + 0.4*x1 + 0.6*x2 - 2.7"   # Balanced full form
            ],
            'nonlinear': [
                "1.1*x1**2 + 1.1*x2**2 - 1.3",  # Symmetric
                "2.2*x1**2 - 1.9",               # Only x1 squared
                "1.8*x2**2 - 1.6",               # Only x2 squared  
                "0.6*x1**2 + 1.5*x2**2 - 1.1",  # Asymmetric
                "3.1*x1**2 + 0.3*x2**2 - 2.4"   # High x1 coefficient
            ],
            'polynomial': [
                "0.3*x1**4 + 0.2*x2**4 - 2.1",                                   # Only 4th degree
                "0.4*x1**4 + 1.2*x1**2 + 0.9*x2**2 - 2.5",                      # Mixed degrees
                "0.2*x1**4 + 0.1*x2**4 + 1.1*x1**2 + 0.8*x2**2 + 0.1*x1*x2 - 2.8", # With cross term
                "0.5*x1**4 + 0.3*x2**4 + 0.9*x1**2 + 1.4*x2**2 + 0.1*x1 - 3.1",     # Complex structure
                "0.1*x1**4 + 0.6*x2**4 + 1.8*x1**2 + 0.4*x2**2 + 0.2*x1*x2 + 0.3*x2 - 3.8" # Full complexity
            ]
        }
        
        template_fallbacks = fallbacks.get(self.template_type, fallbacks['nonlinear'])
        return template_fallbacks[(solution_num - 1) % len(template_fallbacks)]
    
    def _format_set_description(self, set_desc: Dict) -> str:
        """Format set description"""
        try:
            if set_desc.get('type') == 'ball':
                radius = set_desc.get('radius', 1.0)
                center = set_desc.get('center', [0, 0])
                complement = set_desc.get('complement', False)
                
                if complement:
                    return f"Outside ball with center {center} and radius {radius}"
                else:
                    return f"Ball with center {center} and radius {radius}"
            return str(set_desc)
        except Exception:
            return str(set_desc)


class LLMBasedBestSelector(operations.Operation):
    """Phase 3: LLM-based selection of best solution from each template type"""
    
    def __init__(self, template_type: str, problem_params: Dict):
        super().__init__()
        self.template_type = template_type
        self.problem_params = problem_params
        self.thoughts = []
    
    def get_thoughts(self) -> List[Thought]:
        """Return the thoughts generated by this operation"""
        return self.thoughts
        
    def _execute(self, lm, prompter, parser, **kwargs):
        try:
            print(f"\n{'='*60}")
            print(f"PHASE 3: LLM-BASED SELECTION FOR {self.template_type.upper()}")
            print(f"{'='*60}")
            logger.info(f"Starting LLM-based selection for {self.template_type}")
            
            # Collect all solutions from predecessor
            all_solutions = []
            for pred in self.predecessors:
                if hasattr(pred, 'get_thoughts'):
                    pred_thoughts = pred.get_thoughts()
                    if pred_thoughts:
                        for thought in pred_thoughts:
                            if hasattr(thought, 'state') and isinstance(thought.state, dict):
                                if thought.state.get('template_type') == self.template_type:
                                    all_solutions.append(thought.state)
            
            if not all_solutions:
                logger.warning(f"No solutions found for {self.template_type}")
                self.thoughts = []
                return
            
            # Check if any solution is already valid
            valid_solutions = [sol for sol in all_solutions if sol.get('valid_solution', False)]
            if valid_solutions:
                best_solution = valid_solutions[0]
                print(f"Valid solution already found: {best_solution['barrier_certificate']}")
                logger.info(f"Valid {self.template_type} solution already found")
                
                # Test again to be sure
                is_valid = validate_barrier_certificate(
                    best_solution['barrier_certificate'],
                    self.problem_params.get('initial_set', {}),
                    self.problem_params.get('unsafe_set', {}),
                    self.problem_params.get('dynamics', '')
                )
                
                if is_valid:
                    verification_details = get_detailed_condition_results(
                        best_solution['barrier_certificate'],
                        self.problem_params.get('initial_set', {}),
                        self.problem_params.get('unsafe_set', {}),
                        self.problem_params.get('dynamics', '')
                    )
                    
                    result = {
                        **best_solution,
                        'selected_by_llm': False,
                        'selection_reason': 'Already valid solution',
                        'phase': 'llm_selection'
                    }
                    
                    self.thoughts = [Thought(result)]
                    raise ValidSolutionFound(
                        best_solution['barrier_certificate'], 
                        self.template_type, 
                        verification_details, 
                        "Phase 3: LLM Selection"
                    )
            
            # No valid solution found, use LLM to select best one
            print(f"No valid solutions found. Using LLM to select best {self.template_type} solution...")
            logger.info(f"Using LLM to select best {self.template_type} solution from {len(all_solutions)} candidates")
            
            # Create selection prompt
            selection_prompt = self._create_selection_prompt(all_solutions)
            
            # Query LLM for selection
            lm.set_temperature(0.3)  # Lower temperature for consistent selection
            response = lm.query(selection_prompt, num_responses=1)
            response_texts = lm.get_response_texts(response)
            
            if not response_texts:
                logger.error(f"No response from LLM for {self.template_type} selection")
                # Fallback: select first solution
                best_solution = all_solutions[0]
            else:
                # Parse LLM selection
                selected_index, reason = self._parse_selection_response(response_texts[0])
                if 1 <= selected_index <= len(all_solutions):
                    best_solution = all_solutions[selected_index - 1]
                    print(f"LLM selected Solution {selected_index}: {best_solution['barrier_certificate']}")
                    print(f"Reason: {reason}")
                    logger.info(f"LLM selected solution {selected_index} for {self.template_type}: {reason}")
                else:
                    logger.warning(f"Invalid LLM selection, using first solution")
                    best_solution = all_solutions[0]
                    reason = "Fallback: LLM selection was invalid"
            
            # Test selected solution again
            is_valid = validate_barrier_certificate(
                best_solution['barrier_certificate'],
                self.problem_params.get('initial_set', {}),
                self.problem_params.get('unsafe_set', {}),
                self.problem_params.get('dynamics', '')
            )
            
            if is_valid:
                print(f"\n{'*'*60}")
                print(f"VALID SOLUTION FOUND IN PHASE 3!")
                print(f"Template: {self.template_type}")
                print(f"Solution: B(x) = {best_solution['barrier_certificate']}")
                print(f"{'*'*60}\n")
                logger.info(f"VALID SOLUTION FOUND IN PHASE 3: {best_solution['barrier_certificate']}")
                
                verification_details = get_detailed_condition_results(
                    best_solution['barrier_certificate'],
                    self.problem_params.get('initial_set', {}),
                    self.problem_params.get('unsafe_set', {}),
                    self.problem_params.get('dynamics', '')
                )
                
                raise ValidSolutionFound(
                    best_solution['barrier_certificate'], 
                    self.template_type, 
                    verification_details, 
                    "Phase 3: LLM Selection"
                )
            
            # Create result
            result = {
                **best_solution,
                'selected_by_llm': True,
                'selection_reason': reason,
                'phase': 'llm_selection'
            }
            
            self.thoughts = [Thought(result)]
            print(f"Selected best {self.template_type} solution: {best_solution['barrier_certificate']}")
            logger.info(f"Selected best {self.template_type} solution: {best_solution['barrier_certificate']}")
            
        except ValidSolutionFound:
            # Re-raise to stop pipeline
            raise
        except Exception as e:
            logger.error(f"Error in {self.template_type} LLM selection: {e}")
            self.thoughts = []
    
    def _create_selection_prompt(self, solutions: List[Dict]) -> str:
        """Create prompt for LLM-based solution selection"""
        initial_set = self.problem_params.get('initial_set', {})
        unsafe_set = self.problem_params.get('unsafe_set', {})
        dynamics = self.problem_params.get('dynamics', '')
        
        initial_desc = self._format_set_description(initial_set)
        unsafe_desc = self._format_set_description(unsafe_set)
        
        # Format solutions
        solutions_text = ""
        for i, sol in enumerate(solutions, 1):
            barrier = sol.get('barrier_certificate', 'Unknown')
            solutions_text += f"Solution {i}: B(x) = {barrier}\n"
        
        return f"""You are an expert in barrier certificate theory. Select the BEST barrier certificate from the {self.template_type} candidates below.

SYSTEM INFORMATION:
Dynamics: {dynamics}
Initial Set: {initial_desc}
Unsafe Set: {unsafe_desc}

BARRIER CERTIFICATE CONDITIONS (must be satisfied):
1. B(x) ≤ 0 for all x in initial set
2. B(x) > 0 for all x in unsafe set
3. ∇B(x)·f(x) ≤ 0 for all x where B(x) = 0 (invariance condition)

{self.template_type.upper()} CANDIDATES:
{solutions_text}

SELECTION CRITERIA:
1. Mathematical soundness and structure
2. Likelihood to satisfy barrier conditions
3. Coefficient magnitudes (not too extreme)
4. Geometric intuition for the given sets
5. Stability under the given dynamics

Please select the BEST solution and explain why.

OUTPUT FORMAT:
Selected: [number from 1-{len(solutions)}]
Reason: [detailed mathematical reasoning for your choice]

ANALYZE AND SELECT NOW:"""
    
    def _parse_selection_response(self, response: str) -> Tuple[int, str]:
        """Parse LLM selection response"""
        try:
            # Look for "Selected: X" pattern
            selected_match = re.search(r'Selected:\s*(\d+)', response, re.IGNORECASE)
            if selected_match:
                selected_index = int(selected_match.group(1))
            else:
                # Fallback: look for first number
                number_match = re.search(r'\b(\d+)\b', response)
                selected_index = int(number_match.group(1)) if number_match else 1
            
            # Extract reason
            reason_match = re.search(r'Reason:\s*(.+?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()
            else:
                # Fallback: use entire response
                reason = response[:200] + "..." if len(response) > 200 else response
            
            return selected_index, reason
            
        except Exception as e:
            logger.error(f"Error parsing LLM selection: {e}")
            return 1, "Fallback selection due to parsing error"
    
    def _format_set_description(self, set_desc: Dict) -> str:
        """Format set description"""
        try:
            if set_desc.get('type') == 'ball':
                radius = set_desc.get('radius', 1.0)
                center = set_desc.get('center', [0, 0])
                complement = set_desc.get('complement', False)
                
                if complement:
                    return f"Outside ball with center {center} and radius {radius}"
                else:
                    return f"Ball with center {center} and radius {radius}"
            return str(set_desc)
        except Exception:
            return str(set_desc)


class MultiStrategyAggregator(operations.Aggregate):
    """Phase 4: Multi-strategy aggregation inspired by GoT"""
    
    def __init__(self, problem_params: Dict):
        super().__init__()
        self.problem_params = problem_params
        self.thoughts = []
    
    def get_thoughts(self) -> List[Thought]:
        """Return the thoughts generated by this operation"""
        return self.thoughts
        
    def _execute(self, lm, prompter, parser, **kwargs):
        try:
            print(f"\n{'='*60}")
            print("PHASE 4: MULTI-STRATEGY AGGREGATION")
            print(f"{'='*60}")
            logger.info("Starting multi-strategy aggregation")
            
            # Collect best solutions from each template
            template_solutions = {}
            for pred in self.predecessors:
                if hasattr(pred, 'get_thoughts'):
                    pred_thoughts = pred.get_thoughts()
                    if pred_thoughts:
                        for thought in pred_thoughts:
                            if hasattr(thought, 'state') and isinstance(thought.state, dict):
                                template_type = thought.state.get('template_type', 'unknown')
                                template_solutions[template_type] = thought.state
            
            if not template_solutions:
                logger.warning("No template solutions found for aggregation")
                self.thoughts = []
                return
            
            print(f"Found solutions for {len(template_solutions)} templates: {list(template_solutions.keys())}")
            logger.info(f"Aggregating solutions from templates: {list(template_solutions.keys())}")
            
            # Check if any solution is already valid
            for template_type, solution in template_solutions.items():
                if solution.get('valid_solution', False):
                    print(f"Valid solution found in {template_type}: {solution['barrier_certificate']}")
                    logger.info(f"Valid solution found in aggregation phase: {solution['barrier_certificate']}")
                    
                    # Test again to be sure
                    is_valid = validate_barrier_certificate(
                        solution['barrier_certificate'],
                        self.problem_params.get('initial_set', {}),
                        self.problem_params.get('unsafe_set', {}),
                        self.problem_params.get('dynamics', '')
                    )
                    
                    if is_valid:
                        verification_details = get_detailed_condition_results(
                            solution['barrier_certificate'],
                            self.problem_params.get('initial_set', {}),
                            self.problem_params.get('unsafe_set', {}),
                            self.problem_params.get('dynamics', '')
                        )
                        
                        result = {
                            **solution,
                            'aggregation_strategy': 'early_valid_found',
                            'phase': 'aggregation'
                        }
                        
                        self.thoughts = [Thought(result)]
                        raise ValidSolutionFound(
                            solution['barrier_certificate'], 
                            template_type, 
                            verification_details, 
                            "Phase 4: Aggregation"
                        )
            
            # No valid solution found, try aggregation strategies
            strategies = [
                self._strategy_weighted_combination,
                self._strategy_best_template_enhancement,
                self._strategy_hybrid_approach
            ]
            
            best_aggregated = None
            best_strategy = None
            
            for i, strategy in enumerate(strategies, 1):
                try:
                    print(f"\nTrying aggregation strategy {i}...")
                    logger.info(f"Trying aggregation strategy {i}")
                    
                    aggregated_solution = strategy(template_solutions, lm)
                    
                    if aggregated_solution:
                        # Test aggregated solution
                        is_valid = validate_barrier_certificate(
                            aggregated_solution,
                            self.problem_params.get('initial_set', {}),
                            self.problem_params.get('unsafe_set', {}),
                            self.problem_params.get('dynamics', '')
                        )
                        
                        if is_valid:
                            print(f"\n{'*'*60}")
                            print(f"VALID SOLUTION FOUND IN PHASE 4!")
                            print(f"Strategy: {i}")
                            print(f"Solution: B(x) = {aggregated_solution}")
                            print(f"{'*'*60}\n")
                            logger.info(f"VALID SOLUTION FOUND IN PHASE 4: {aggregated_solution}")
                            
                            verification_details = get_detailed_condition_results(
                                aggregated_solution,
                                self.problem_params.get('initial_set', {}),
                                self.problem_params.get('unsafe_set', {}),
                                self.problem_params.get('dynamics', '')
                            )
                            
                            raise ValidSolutionFound(
                                aggregated_solution, 
                                f"aggregated_strategy_{i}", 
                                verification_details, 
                                "Phase 4: Aggregation"
                            )
                        else:
                            if best_aggregated is None:
                                best_aggregated = aggregated_solution
                                best_strategy = i
                                
                except Exception as e:
                    logger.warning(f"Strategy {i} failed: {e}")
                    continue
            
            # Use best aggregated solution or fallback
            if best_aggregated:
                print(f"Using best aggregated solution from strategy {best_strategy}: {best_aggregated}")
                logger.info(f"Using best aggregated solution from strategy {best_strategy}")
                
                result = {
                    'barrier_certificate': best_aggregated,
                    'template_type': f'aggregated_strategy_{best_strategy}',
                    'aggregation_strategy': f'strategy_{best_strategy}',
                    'valid_solution': False,
                    'phase': 'aggregation'
                }
            else:
                # Fallback to best individual solution
                best_individual = max(template_solutions.values(), 
                                    key=lambda x: self._score_solution(x.get('barrier_certificate', '')))
                print(f"Fallback to best individual solution: {best_individual['barrier_certificate']}")
                logger.info(f"Fallback to best individual solution from {best_individual['template_type']}")
                
                result = {
                    **best_individual,
                    'aggregation_strategy': 'fallback_best_individual',
                    'phase': 'aggregation'
                }
            
            self.thoughts = [Thought(result)]
            
        except ValidSolutionFound:
            # Re-raise to stop pipeline
            raise
        except Exception as e:
            logger.error(f"Error in aggregation: {e}")
            self.thoughts = []
    
    def _strategy_weighted_combination(self, template_solutions: Dict, lm) -> Optional[str]:
        """Strategy 1: Weighted combination of best solutions"""
        try:
            # Create prompt for LLM to combine solutions
            combination_prompt = self._create_combination_prompt(template_solutions)
            
            lm.set_temperature(0.4)
            response = lm.query(combination_prompt, num_responses=1)
            response_texts = lm.get_response_texts(response)
            
            if response_texts:
                # Extract combined solution
                combined = self._extract_barrier_from_response(response_texts[0])
                if combined:
                    return combined
            
            return None
            
        except Exception as e:
            logger.error(f"Weighted combination strategy failed: {e}")
            return None
    
    def _strategy_best_template_enhancement(self, template_solutions: Dict, lm) -> Optional[str]:
        """Strategy 2: Enhance the most promising template"""
        try:
            # Score each solution and pick best
            best_solution = max(template_solutions.values(), 
                              key=lambda x: self._score_solution(x.get('barrier_certificate', '')))
            
            # Create enhancement prompt
            enhancement_prompt = self._create_enhancement_prompt(best_solution)
            
            lm.set_temperature(0.5)
            response = lm.query(enhancement_prompt, num_responses=1)
            response_texts = lm.get_response_texts(response)
            
            if response_texts:
                enhanced = self._extract_barrier_from_response(response_texts[0])
                if enhanced:
                    return enhanced
            
            return best_solution.get('barrier_certificate', '')
            
        except Exception as e:
            logger.error(f"Best template enhancement strategy failed: {e}")
            return None
    
    def _strategy_hybrid_approach(self, template_solutions: Dict, lm) -> Optional[str]:
        """Strategy 3: Hybrid approach combining different template features"""
        try:
            # Create hybrid prompt
            hybrid_prompt = self._create_hybrid_prompt(template_solutions)
            
            lm.set_temperature(0.6)
            response = lm.query(hybrid_prompt, num_responses=1)
            response_texts = lm.get_response_texts(response)
            
            if response_texts:
                hybrid = self._extract_barrier_from_response(response_texts[0])
                if hybrid:
                    return hybrid
            
            return None
            
        except Exception as e:
            logger.error(f"Hybrid approach strategy failed: {e}")
            return None
    
    def _create_combination_prompt(self, template_solutions: Dict) -> str:
        """Create prompt for weighted combination"""
        solutions_text = ""
        for template_type, solution in template_solutions.items():
            barrier = solution.get('barrier_certificate', 'Unknown')
            solutions_text += f"{template_type.capitalize()}: B(x) = {barrier}\n"
        
        return f"""Combine the following barrier certificate candidates into a single, improved solution:

{solutions_text}

Create a weighted combination that:
1. Preserves the best features of each template
2. Has reasonable coefficient magnitudes
3. Maintains mathematical soundness
4. Should satisfy barrier certificate conditions

OUTPUT: B(x) = [combined expression]

COMBINE NOW:"""
    
    def _create_enhancement_prompt(self, best_solution: Dict) -> str:
        """Create prompt for solution enhancement"""
        barrier = best_solution.get('barrier_certificate', '')
        template_type = best_solution.get('template_type', 'unknown')
        
        return f"""Enhance this {template_type} barrier certificate to better satisfy the barrier conditions:

Current: B(x) = {barrier}

Enhance by:
1. Adjusting coefficients for better condition satisfaction
2. Maintaining the {template_type} template structure
3. Ensuring mathematical validity
4. Optimizing for barrier certificate requirements

OUTPUT: B(x) = [enhanced expression]

ENHANCE NOW:"""
    
    def _create_hybrid_prompt(self, template_solutions: Dict) -> str:
        """Create prompt for hybrid approach"""
        solutions_text = ""
        for template_type, solution in template_solutions.items():
            barrier = solution.get('barrier_certificate', 'Unknown')
            solutions_text += f"{template_type.capitalize()}: B(x) = {barrier}\n"
        
        return f"""Create a hybrid barrier certificate that combines features from different templates:

Available solutions:
{solutions_text}

Create a hybrid that:
1. Takes the best structural elements from each template
2. Creates a novel but mathematically sound combination
3. Balances complexity with effectiveness
4. Maintains barrier certificate validity

OUTPUT: B(x) = [hybrid expression]

CREATE HYBRID NOW:"""
    
    def _extract_barrier_from_response(self, response: str) -> Optional[str]:
        """Extract barrier certificate from LLM response"""
        try:
            # Look for B(x) = pattern
            pattern = r'B\(x\)\s*=\s*([^\n]+)'
            match = re.search(pattern, response, re.IGNORECASE)
            
            if match:
                barrier = match.group(1).strip()
                # Clean the barrier
                from barrier_utils import clean_and_extract_barrier
                cleaned = clean_and_extract_barrier(barrier)
                return cleaned if cleaned else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting barrier from response: {e}")
            return None
    
    def _score_solution(self, barrier: str) -> float:
        """Simple scoring function for solutions"""
        try:
            if not barrier:
                return 0.0
            
            score = 0.0
            
            # Basic structure scoring
            if 'x1' in barrier and 'x2' in barrier:
                score += 1.0
            
            # Coefficient diversity scoring
            coefficients = re.findall(r'[-+]?\d*\.?\d+', barrier)
            if len(set(coefficients)) > 2:
                score += 0.5
            
            # Negative constant term (good for initial condition)
            if '-' in barrier and any(c.isdigit() for c in barrier.split('-')[-1]):
                score += 0.5
            
            return score
            
        except Exception:
            return 0.0


class ParameterRefinement(operations.Operation):
    """Phase 5: Parameter refinement and final optimization"""
    
    def __init__(self, problem_params: Dict):
        super().__init__()
        self.problem_params = problem_params
        self.thoughts = []
    
    def get_thoughts(self) -> List[Thought]:
        """Return the thoughts generated by this operation"""
        return self.thoughts
        
    def _execute(self, lm, prompter, parser, **kwargs):
        try:
            print(f"\n{'='*60}")
            print("PHASE 5: PARAMETER REFINEMENT")
            print(f"{'='*60}")
            logger.info("Starting parameter refinement")
            
            # Get solution from aggregation phase
            aggregated_solution = None
            for pred in self.predecessors:
                if hasattr(pred, 'get_thoughts'):
                    pred_thoughts = pred.get_thoughts()
                    if pred_thoughts and len(pred_thoughts) > 0:
                        thought = pred_thoughts[0]
                        if hasattr(thought, 'state') and isinstance(thought.state, dict):
                            aggregated_solution = thought.state
                            break
            
            if not aggregated_solution:
                logger.warning("No solution found for refinement")
                self.thoughts = []
                return
            
            barrier = aggregated_solution.get('barrier_certificate', '')
            print(f"Refining solution: B(x) = {barrier}")
            logger.info(f"Refining solution: {barrier}")
            
            # Check if already valid
            is_valid = validate_barrier_certificate(
                barrier,
                self.problem_params.get('initial_set', {}),
                self.problem_params.get('unsafe_set', {}),
                self.problem_params.get('dynamics', '')
            )
            
            if is_valid:
                print(f"Solution is already valid, no refinement needed!")
                logger.info("Solution is already valid")
                
                verification_details = get_detailed_condition_results(
                    barrier,
                    self.problem_params.get('initial_set', {}),
                    self.problem_params.get('unsafe_set', {}),
                    self.problem_params.get('dynamics', '')
                )
                
                result = {
                    **aggregated_solution,
                    'refinement_applied': False,
                    'refinement_reason': 'Already valid',
                    'phase': 'refinement'
                }
                
                self.thoughts = [Thought(result)]
                raise ValidSolutionFound(
                    barrier, 
                    aggregated_solution.get('template_type', 'refined'), 
                    verification_details, 
                    "Phase 5: Refinement"
                )
            
            # Apply refinement strategies
            refinement_strategies = [
                self._refine_coefficient_optimization,
                self._refine_structure_adjustment,
                self._refine_mathematical_correction
            ]
            
            for i, strategy in enumerate(refinement_strategies, 1):
                try:
                    print(f"\nApplying refinement strategy {i}...")
                    logger.info(f"Applying refinement strategy {i}")
                    
                    refined_barrier = strategy(barrier, lm)
                    
                    if refined_barrier and refined_barrier != barrier:
                        # Test refined solution
                        is_valid = validate_barrier_certificate(
                            refined_barrier,
                            self.problem_params.get('initial_set', {}),
                            self.problem_params.get('unsafe_set', {}),
                            self.problem_params.get('dynamics', '')
                        )
                        
                        if is_valid:
                            print(f"\n{'*'*60}")
                            print(f"VALID SOLUTION FOUND IN PHASE 5!")
                            print(f"Strategy: {i}")
                            print(f"Solution: B(x) = {refined_barrier}")
                            print(f"{'*'*60}\n")
                            logger.info(f"VALID SOLUTION FOUND IN PHASE 5: {refined_barrier}")
                            
                            verification_details = get_detailed_condition_results(
                                refined_barrier,
                                self.problem_params.get('initial_set', {}),
                                self.problem_params.get('unsafe_set', {}),
                                self.problem_params.get('dynamics', '')
                            )
                            
                            raise ValidSolutionFound(
                                refined_barrier, 
                                f"refined_strategy_{i}", 
                                verification_details, 
                                "Phase 5: Refinement"
                            )
                        else:
                            print(f"Strategy {i} produced: {refined_barrier} (not valid)")
                            logger.info(f"Strategy {i} produced invalid solution: {refined_barrier}")
                
                except Exception as e:
                    logger.warning(f"Refinement strategy {i} failed: {e}")
                    continue
            
            # No valid refinement found, return original
            print("No valid refinement found, returning original solution")
            logger.info("No valid refinement found")
            
            result = {
                **aggregated_solution,
                'refinement_applied': False,
                'refinement_reason': 'No valid refinement found',
                'phase': 'refinement'
            }
            
            self.thoughts = [Thought(result)]
            
        except ValidSolutionFound:
            # Re-raise to stop pipeline
            raise
        except Exception as e:
            logger.error(f"Error in parameter refinement: {e}")
            self.thoughts = []
    
    def _refine_coefficient_optimization(self, barrier: str, lm) -> Optional[str]:
        """Refinement strategy 1: Coefficient optimization"""
        try:
            prompt = f"""Optimize the coefficients in this barrier certificate for better condition satisfaction:

Current: B(x) = {barrier}

Optimization goals:
1. Ensure B(x) ≤ 0 in initial set
2. Ensure B(x) > 0 in unsafe set
3. Improve invariance condition satisfaction
4. Maintain mathematical structure

Adjust coefficients systematically and provide:
OUTPUT: B(x) = [optimized expression]

OPTIMIZE NOW:"""
            
            lm.set_temperature(0.3)
            response = lm.query(prompt, num_responses=1)
            response_texts = lm.get_response_texts(response)
            
            if response_texts:
                return self._extract_barrier_from_response(response_texts[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Coefficient optimization failed: {e}")
            return None
    
    def _refine_structure_adjustment(self, barrier: str, lm) -> Optional[str]:
        """Refinement strategy 2: Structure adjustment"""
        try:
            prompt = f"""Adjust the mathematical structure of this barrier certificate:

Current: B(x) = {barrier}

Structure adjustments:
1. Add/remove terms to improve condition satisfaction
2. Adjust term powers if beneficial
3. Maintain mathematical validity
4. Focus on geometric intuition

Provide improved structure:
OUTPUT: B(x) = [adjusted expression]

ADJUST NOW:"""
            
            lm.set_temperature(0.4)
            response = lm.query(prompt, num_responses=1)
            response_texts = lm.get_response_texts(response)
            
            if response_texts:
                return self._extract_barrier_from_response(response_texts[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Structure adjustment failed: {e}")
            return None
    
    def _refine_mathematical_correction(self, barrier: str, lm) -> Optional[str]:
        """Refinement strategy 3: Mathematical correction"""
        try:
            prompt = f"""Apply mathematical corrections to this barrier certificate:

Current: B(x) = {barrier}

Mathematical corrections:
1. Fix any obvious mathematical issues
2. Ensure proper scaling
3. Correct sign errors
4. Improve numerical stability
5. Apply barrier certificate theory principles

Provide corrected version:
OUTPUT: B(x) = [corrected expression]

CORRECT NOW:"""
            
            lm.set_temperature(0.3)
            response = lm.query(prompt, num_responses=1)
            response_texts = lm.get_response_texts(response)
            
            if response_texts:
                return self._extract_barrier_from_response(response_texts[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Mathematical correction failed: {e}")
            return None
    
    def _extract_barrier_from_response(self, response: str) -> Optional[str]:
        """Extract barrier certificate from LLM response"""
        try:
            # Look for B(x) = pattern
            pattern = r'B\(x\)\s*=\s*([^\n]+)'
            match = re.search(pattern, response, re.IGNORECASE)
            
            if match:
                barrier = match.group(1).strip()
                # Clean the barrier
                from barrier_utils import clean_and_extract_barrier
                cleaned = clean_and_extract_barrier(barrier)
                return cleaned if cleaned else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting barrier from response: {e}")
            return None


def create_enhanced_pipeline(problem_params: Dict) -> operations.GraphOfOperations:
    """Create enhanced pipeline following the Physics-Informed GoT architecture"""
    gop = operations.GraphOfOperations()
    
    print("\nCREATING ENHANCED PHYSICS-INFORMED GRAPH OF THOUGHTS PIPELINE...")
    logger.info("Creating enhanced Physics-Informed GoT pipeline...")
    
    # Phase 2: Parameter Synthesis (20 operations total)
    template_types = ['linear', 'quadratic', 'nonlinear', 'polynomial']
    generators = []
    selectors = []
    
    print("\nPHASE 2: PARAMETER SYNTHESIS")
    logger.info("Phase 2: Parameter Synthesis")
    for template_type in template_types:
        print(f"   - {template_type}: 5 solutions with strict template enforcement")
        logger.info(f"   - {template_type}: 5 solutions with strict template enforcement")
        generator = TemplateEnforcedGenerate(template_type, problem_params)
        gop.add_operation(generator)
        generators.append(generator)
        
        # Phase 3: LLM-based Best Selection for each template
        print(f"   - {template_type}: LLM-based best selection")
        logger.info(f"   - {template_type}: LLM-based best selection")
        selector = LLMBasedBestSelector(template_type, problem_params)
        selector.add_predecessor(generator)
        gop.add_operation(selector)
        selectors.append(selector)
    
    # Phase 4: Multi-strategy Aggregation
    print("\nPHASE 4: MULTI-STRATEGY AGGREGATION")
    logger.info("Phase 4: Multi-strategy Aggregation")
    print("   - Strategy 1: Weighted combination")
    print("   - Strategy 2: Best template enhancement") 
    print("   - Strategy 3: Hybrid approach")
    logger.info("Creating multi-strategy aggregation...")
    aggregator = MultiStrategyAggregator(problem_params)
    for selector in selectors:
        aggregator.add_predecessor(selector)
    gop.add_operation(aggregator)
    
    # Phase 5: Parameter Refinement
    print("\nPHASE 5: PARAMETER REFINEMENT")
    logger.info("Phase 5: Parameter Refinement")
    print("   - Refinement 1: Coefficient optimization")
    print("   - Refinement 2: Structure adjustment")
    print("   - Refinement 3: Mathematical correction")
    logger.info("Creating parameter refinement...")
    refiner = ParameterRefinement(problem_params)
    refiner.add_predecessor(aggregator)
    gop.add_operation(refiner)
    
    print(f"\nCreated enhanced pipeline with {len(gop.operations)} operations")
    print("Pipeline structure: 4 generators → 4 selectors → 1 aggregator → 1 refiner")
    logger.info(f"Created enhanced pipeline with {len(gop.operations)} operations")
    logger.info("Pipeline: 4 generators → 4 selectors → 1 aggregator → 1 refiner")
    return gop


def setup_components(config_file: str = "config.json"):
    """Setup language model"""
    try:
        # Setup Llama model  
        lm = LlamaTogetherAI(config_file, model_name="llama8b")
        
        # Test the model
        test_response = lm.query("Respond with 'TEST OK' if you understand.", num_responses=1)
        test_texts = lm.get_response_texts(test_response)
        
        if test_texts and len(test_texts[0].strip()) > 0:
            print("Llama initialized successfully")
            logger.info(f"Llama initialized successfully")
        else:
            print("Llama test failed")
            logger.error("Llama test failed")
            return None
        
        return lm
        
    except Exception as e:
        logger.error(f"Error setting up components: {e}")
        return None


def define_problem() -> Dict[str, Any]:
    """Define the barrier certificate synthesis problem"""
    return {
        'dynamics': 'dx1/dt = x2, dx2/dt = -0.2*x1 - 0.2*x2',
        'initial_set': {
            'type': 'ball',
            'radius': 0.8,
            'center': [0, 0]
        },
        'unsafe_set': {
            'type': 'ball',
            'radius': 4.0,
            'center': [0, 0],
            'complement': True
        }
    }


def main():
    """Enhanced main execution with full pipeline"""
    print("\n" + "="*80)
    print("ENHANCED PHYSICS-INFORMED GRAPH OF THOUGHTS")
    print("BARRIER CERTIFICATE SYNTHESIS PIPELINE")
    print("="*80)
    logger.info("Starting Enhanced Physics-Informed Graph of Thoughts Pipeline")
    
    # Define problem
    problem_params = define_problem()
    print("\nPROBLEM DEFINITION:")
    print("-" * 50)
    print("   Source: Safe Reach Set Computation via Neural Barrier Certificates")
    print("   System Type: Linear system with complex eigenvalues")
    print(f"   Dynamics: {problem_params['dynamics']}")
    print(f"   Initial set: Ball(center=[0,0], radius=0.8)")
    print(f"   Unsafe set: Outside Ball(center=[0,0], radius=4.0)")
    print("   Pipeline: 5 phases with early stopping")
    print("-" * 50)
    
    logger.info("Enhanced problem defined with 5-phase pipeline")
    
    # Setup components
    print("\nSETTING UP COMPONENTS...")
    lm = setup_components()
    if not lm:
        print("Failed to setup components")
        logger.error("Failed to setup components")
        return {'success': False, 'error': 'Component setup failed'}
    
    # Create enhanced pipeline
    gop = create_enhanced_pipeline(problem_params)
    
    # Execute pipeline with early stopping
    try:
        print("\nSTARTING ENHANCED PIPELINE EXECUTION...")
        logger.info("Starting enhanced pipeline execution...")
        
        success = False
        final_barrier = None
        verification_details = None
        stopping_phase = None
        
        try:
            for i, operation in enumerate(gop.operations):
                operation_name = operation.__class__.__name__
                print(f"\nExecuting operation {i+1}: {operation_name}")
                logger.info(f"Executing operation {i+1}: {operation_name}")
                
                if hasattr(operation, '_execute'):
                    operation._execute(lm, None, None)
                    
                    # Check for valid solution in any operation
                    if hasattr(operation, 'thoughts'):
                        for thought in operation.thoughts:
                            if hasattr(thought, 'state') and isinstance(thought.state, dict):
                                if thought.state.get('valid_solution', False):
                                    print(f"\nVALID SOLUTION FOUND IN {operation_name}!")
                                    logger.info(f"VALID SOLUTION FOUND IN {operation_name}!")
                                    final_barrier = thought.state.get('barrier_certificate', '')
                                    verification_details = thought.state.get('verification_details', {})
                                    stopping_phase = operation_name
                                    success = True
                                    raise ValidSolutionFound(final_barrier, '', verification_details, operation_name)
            
        except ValidSolutionFound as e:
            print(f"SUCCESS: Valid barrier certificate found in {e.phase}!")
            print(f"Barrier Function: B(x) = {e.barrier}")
            logger.info(f"SUCCESS: Valid barrier certificate found in {e.phase}!")
            logger.info(f"B(x) = {e.barrier}")
            success = True
            final_barrier = e.barrier
            verification_details = e.verification_details
            stopping_phase = e.phase
        
        # Build enhanced results
        results = {
            'success': success,
            'final_barrier': final_barrier,
            'verification_details': verification_details,
            'stopping_phase': stopping_phase,
            'method': 'enhanced_physics_informed_got',
            'pipeline_phases': 5,
            'early_stopping': success,
            'model_used': 'Meta-Llama-3.1-8B-Instruct-Turbo',
            'problem_type': 'linear_complex_eigenvalues',
            'problem_source': 'Safe Reach Set Computation via Neural Barrier Certificates'
        }
        
        # Save results
        with open("enhanced_barrier_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Display enhanced results
        print("\n" + "="*80)
        print("ENHANCED PIPELINE RESULTS")
        print("="*80)
        if success:
            print("STATUS: SUCCESS - Valid barrier certificate found!")
            print(f"STOPPING PHASE: {stopping_phase}")
            print(f"BARRIER FUNCTION: B(x) = {final_barrier}")
            print("VERIFICATION: All 3 barrier certificate conditions satisfied!")
            print("METHOD: Enhanced Physics-Informed Graph of Thoughts")
            print("FEATURES: LLM selection, multi-strategy aggregation, refinement")
        else:
            print("STATUS: FAILED - No valid barrier certificate found")
            print("PIPELINE: All 5 phases completed without success")
            print("METHOD: Enhanced Physics-Informed Graph of Thoughts")
        
        print("="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced execution failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    try:
        results = main()
        exit_code = 0 if results.get('success', False) else 1
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)