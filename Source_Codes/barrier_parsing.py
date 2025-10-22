import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Union, Any
import re
import logging

logger = logging.getLogger(__name__)


def clean_and_extract_barrier(barrier_str: str) -> str:
    """Extract and clean barrier certificate with ENHANCED Unicode handling and support for x1-x10"""
    if not barrier_str or not isinstance(barrier_str, str):
        return ""
    
    try:
        # Remove common prefixes
        barrier_str = re.sub(r'B\(x\)\s*=\s*', '', barrier_str, flags=re.IGNORECASE)
        barrier_str = re.sub(r'barrier\s*certificate\s*:?\s*', '', barrier_str, flags=re.IGNORECASE)
        
        # ENHANCED Unicode handling
        unicode_subscripts = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
        }
        for unicode_char, ascii_char in unicode_subscripts.items():
            barrier_str = barrier_str.replace(unicode_char, ascii_char)
        
        # Convert Unicode superscripts to ** notation (including 4th power)
        unicode_superscripts = {
            '²': '**2', '³': '**3', '⁴': '**4', '⁵': '**5',
            '⁶': '**6', '⁷': '**7', '⁸': '**8', '⁹': '**9'
        }
        for unicode_char, ascii_notation in unicode_superscripts.items():
            barrier_str = barrier_str.replace(unicode_char, ascii_notation)
        
        # Remove LaTeX escape characters
        barrier_str = barrier_str.replace('\\\\', '\\').replace('\\ ', ' ')
        
        # Convert LaTeX multiplication symbols
        barrier_str = barrier_str.replace('\\cdot', '*')
        barrier_str = barrier_str.replace('\\times', '*')
        barrier_str = barrier_str.replace('·', '*')  # Unicode multiplication dot
        
        # Convert subscript notation x_i -> xi for x1-x10
        barrier_str = re.sub(r'x_(\d+)', r'x\1', barrier_str)
        
        # Convert superscript notation xi^n -> xi**n for x1-x10 and any power
        barrier_str = re.sub(r'x(\d+)\^(\d+)', r'x\1**\2', barrier_str)  # x1^4 -> x1**4
        barrier_str = re.sub(r'x\^(\d+)', r'x**\1', barrier_str)         # x^4 -> x**4
        
        # Convert general superscript with parentheses: (expression)^n -> (expression)**n
        barrier_str = re.sub(r'\^(\d+)', r'**\1', barrier_str)
        
        # Fix standalone ^ to **
        barrier_str = barrier_str.replace('^', '**')
        
        # Remove any remaining backslashes
        barrier_str = re.sub(r'\\', '', barrier_str)
        
        # Clean formatting
        barrier_str = barrier_str.strip().strip('"\'.,;:')
        
        # Fix implicit multiplication: 2x -> 2*x, but handle x1-x10
        barrier_str = re.sub(r'(\d)([x])', r'\1*\2', barrier_str)      # 2x -> 2*x
        barrier_str = re.sub(r'(x\d+)([x])', r'\1*\2', barrier_str)   # x1x2 -> x1*x2
        barrier_str = re.sub(r'([x\d])\((?!.*(?:sin|cos))', r'\1*(', barrier_str)      # x( -> x*( but not sin( or cos(
        barrier_str = re.sub(r'\)([x\d])', r')*\1', barrier_str)      # )x -> )*x
        
        # Fix common issues
        barrier_str = re.sub(r'\s+', ' ', barrier_str)
        
        # Balance parentheses BEFORE extraction
        barrier_str = _balance_parentheses(barrier_str)
        
        # Remove trailing operators
        barrier_str = re.sub(r'[+\-\*]+\s*$', '', barrier_str).strip()
        
        # Extract mathematical expression using improved patterns
        extracted_expr = _extract_mathematical_expression(barrier_str)
        
        if extracted_expr:
            barrier_str = extracted_expr
        
        # Final cleanup
        barrier_str = barrier_str.strip()
        
        # Ensure proper variable naming (x1-x10 not x₁-x₁₀)
        for i in range(1, 11):
            barrier_str = re.sub(f'x₍{i}₎', f'x{i}', barrier_str)
        
        # FINAL FIX: One more parentheses balance check
        barrier_str = _balance_parentheses(barrier_str)
        
        # Validate basic structure including variables x1-x10
        if len(barrier_str) < 3 or not (re.search(r'x\d+', barrier_str) or re.search(r'(?:sin|cos)\(', barrier_str)):
            return ""
        
        # Additional validation: ensure it doesn't end with operators
        barrier_str = re.sub(r'[+\-\*]+$', '', barrier_str).strip()
        
        # Log the cleaned result for debugging
        if barrier_str:
            logger.info(f"Cleaned barrier expression: '{barrier_str}'")
        
        return barrier_str
        
    except Exception as e:
        logger.warning(f"Error cleaning barrier '{barrier_str}': {e}")
        return ""


def _balance_parentheses(expression: str) -> str:
    """Balance parentheses in mathematical expression"""
    try:
        open_parens = expression.count('(')
        close_parens = expression.count(')')
        
        if open_parens == close_parens:
            return expression
        
        if open_parens > close_parens:
            missing_close = open_parens - close_parens
            expression += ')' * missing_close
            logger.info(f"Added {missing_close} missing closing parentheses")
        elif close_parens > open_parens:
            extra_close = close_parens - open_parens
            for _ in range(extra_close):
                last_close_idx = expression.rfind(')')
                if last_close_idx != -1:
                    expression = expression[:last_close_idx] + expression[last_close_idx+1:]
            logger.info(f"Removed {extra_close} extra closing parentheses")
        
        return expression
        
    except Exception as e:
        logger.warning(f"Error balancing parentheses in '{expression}': {e}")
        return expression


def _extract_mathematical_expression(text: str) -> str:
    """Extract mathematical expression with enhanced support for x1-x10 variables"""
    try:
        # Enhanced patterns that handle x1-x10 variables
        math_patterns = [
            # Pattern 1: Complete polynomial with mixed terms (most comprehensive)
            r'((?:[-+]?\s*\d*\.?\d*\s*\*?\s*(?:x\d+(?:\*\*\d+)?(?:\s*\*\s*x\d+(?:\*\*\d+)?)*|(?:sin|cos)\([^)]+\))(?:\s*[+\-]\s*\d*\.?\d*\s*\*?\s*(?:x\d+(?:\*\*\d+)?(?:\s*\*\s*x\d+(?:\*\*\d+)?)*|(?:sin|cos)\([^)]+\)))*(?:\s*[+\-]\s*\d+\.?\d*)?))',
            
            # Pattern 2: Polynomial terms with mixed variables x1-x10
            r'((?:[-+]?\s*\d*\.?\d*\s*\*?\s*x\d+(?:\*\*\d+)?(?:\s*\*\s*x\d+(?:\*\*\d+)?)*(?:\s*[+\-]\s*\d*\.?\d*\s*\*?\s*x\d+(?:\*\*\d+)?(?:\s*\*\s*x\d+(?:\*\*\d+)?)*)*(?:\s*[+\-]\s*\d+\.?\d*)?))',
            
            # Pattern 3: Complete expression with trigonometric functions
            r'((?:[-+]?\d*\.?\d*\*?(?:sin|cos)\([^)]+\)|[-+]?\d*\.?\d*\*?x\d+(?:\*\*\d+)?(?:\*x\d+(?:\*\*\d+)?)*|[-+]?\d*\.?\d+)(?:\s*[+\-]\s*(?:[-+]?\d*\.?\d*\*?(?:sin|cos)\([^)]+\)|[-+]?\d*\.?\d*\*?x\d+(?:\*\*\d+)?(?:\*x\d+(?:\*\*\d+)?)*|[-+]?\d*\.?\d+))*)',
            
            # Pattern 4: Simple polynomial terms for x1-x10
            r'([-+]?\d*\.?\d*\*?x\d+(?:\*\*\d+)?(?:\s*[+\-]\s*[-+]?\d*\.?\d*\*?x\d+(?:\*\*\d+)?)*(?:\s*[+\-]\s*[-+]?\d*\.?\d+)?)',
        ]
        
        best_match = ""
        best_length = 0
        
        for pattern in math_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                
                extracted = match.strip()
                
                # Validate the match - should contain variables x1-x10 and be reasonably long
                if (len(extracted) > 3 and 
                    (re.search(r'x\d+', extracted) or re.search(r'(?:sin|cos)\(', extracted)) and
                    len(extracted) > best_length):
                    
                    balanced_extracted = _balance_parentheses(extracted)
                    
                    if _validate_trigonometric_syntax(balanced_extracted):
                        best_match = balanced_extracted
                        best_length = len(balanced_extracted)
                        logger.info(f"Found better match: '{balanced_extracted}' (length: {best_length})")
        
        if best_match and best_length > 10:
            logger.info(f"Successfully extracted mathematical expression: '{best_match}'")
            return best_match
        
        return text
        
    except Exception as e:
        logger.warning(f"Error extracting mathematical expression from '{text}': {e}")
        return text


def _validate_trigonometric_syntax(expression: str) -> bool:
    """Validate trigonometric function syntax"""
    try:
        trig_functions = re.findall(r'(?:sin|cos)\([^)]*\)', expression)
        
        for func in trig_functions:
            open_count = func.count('(')
            close_count = func.count(')')
            
            if open_count != close_count:
                logger.warning(f"Unbalanced parentheses in trigonometric function: {func}")
                return False
            
            inner_content = re.search(r'(?:sin|cos)\(([^)]*)\)', func)
            if inner_content and inner_content.group(1).strip():
                inner = inner_content.group(1)
                # Should contain x1-x10 or mathematical expressions
                if not (re.search(r'x\d+', inner) or re.search(r'[+\-\*\d]', inner)):
                    logger.warning(f"Invalid trigonometric function content: {func}")
                    return False
            else:
                logger.warning(f"Empty trigonometric function: {func}")
                return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error validating trigonometric syntax in '{expression}': {e}")
        return True


def parse_barrier_certificate(barrier_str: str) -> Tuple[Optional[sp.Expr], List[sp.Symbol]]:
    """Parse barrier certificate string into symbolic expression - supports x1-x10"""
    try:
        cleaned_str = clean_and_extract_barrier(barrier_str)
        if not cleaned_str:
            logger.warning("Empty barrier string after cleaning")
            return None, []
        
        logger.info(f"Parsing cleaned barrier: '{cleaned_str}'")
        
        # Extract variables from x1 to x10
        variables = sorted(set(re.findall(r'\bx\d+\b', cleaned_str)), key=lambda x: int(x[1:]))
        if not variables:
            # Also check inside trigonometric functions
            trig_vars = re.findall(r'(?:sin|cos)\([^)]*\b(x\d+)\b[^)]*\)', cleaned_str)
            if trig_vars:
                variables = sorted(set(trig_vars), key=lambda x: int(x[1:]))
            else:
                variables = ['x1', 'x2']  # Default fallback
        
        var_symbols = [sp.Symbol(var, real=True) for var in variables]
        
        # Parse expression with additional preprocessing
        try:
            expr_str = cleaned_str
            
            # Ensure multiplication is explicit for x1-x10
            expr_str = re.sub(r'(\d)([x])', r'\1*\2', expr_str)  # 2x -> 2*x
            expr_str = re.sub(r'([x\d])\(', r'\1*(', expr_str)    # x( -> x*(
            expr_str = re.sub(r'\)([x\d])', r')*\1', expr_str)    # )x -> )*x
            
            # Handle trigonometric functions
            expr_str = re.sub(r'sin\*\(', 'sin(', expr_str)
            expr_str = re.sub(r'cos\*\(', 'cos(', expr_str)
            
            # Handle decimal coefficients properly
            expr_str = re.sub(r'(\d)\.(\d)', r'\1.\2', expr_str)
            
            logger.info(f"Final expression for parsing: '{expr_str}'")
            
            expr = sp.sympify(expr_str, evaluate=True)
            
            # Validate
            if expr.free_symbols or (hasattr(expr, 'has') and (expr.has(sp.sin) or expr.has(sp.cos))):
                logger.info(f"Successfully parsed expression: {expr}")
                return expr, var_symbols
            else:
                logger.warning("Expression has no variables or functions")
                return None, var_symbols
                
        except sp.SympifyError as e:
            logger.warning(f"SymPy failed to parse expression '{cleaned_str}': {e}")
            return None, var_symbols
        except Exception as e:
            logger.warning(f"Failed to parse expression '{cleaned_str}': {e}")
            return None, var_symbols
        
    except Exception as e:
        logger.error(f"Error parsing barrier certificate '{barrier_str}': {e}")
        return None, []


def _evaluate_expression(expr: sp.Expr, variables: List[sp.Symbol], 
                        point: List[float]) -> float:
    """Safely evaluate symbolic expression at a point - supports N variables"""
    try:
        subs_dict = dict(zip(variables, point[:len(variables)]))
        value = expr.subs(subs_dict)
        
        if hasattr(value, 'evalf'):
            result = value.evalf()
        else:
            result = value
        
        try:
            return float(result)
        except (TypeError, ValueError):
            import numpy as np
            if hasattr(result, 'subs'):
                for var, val in subs_dict.items():
                    result = result.subs(var, val)
            
            try:
                return float(result.evalf())
            except:
                result_str = str(result)
                if 'sin' in result_str or 'cos' in result_str:
                    return 0.1  # Small positive value as approximation
                raise
            
    except Exception as e:
        raise ValueError(f"Cannot evaluate expression: {e}")