import sympy as sp
from typing import List, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)


def clean_and_extract_barrier(barrier_str: str) -> str:
    # extract and clean barrier certificate 
    if not barrier_str or not isinstance(barrier_str, str):
        return ""
    
    try:
        # common prefixes
        barrier_str = re.sub(r'B\(x\)\s*=\s*', '', barrier_str, flags=re.IGNORECASE)
        barrier_str = re.sub(r'barrier\s*certificate\s*:?\s*', '', barrier_str, flags=re.IGNORECASE)
        
        # unicode subscripts handling
        unicode_subscripts = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
        }
        for unicode_char, ascii_char in unicode_subscripts.items():
            barrier_str = barrier_str.replace(unicode_char, ascii_char)
        
        # unicode superscripts handling 
        unicode_superscripts = {
            '²': '**2', '³': '**3', '⁴': '**4', '⁵': '**5',
            '⁶': '**6', '⁷': '**7', '⁸': '**8', '⁹': '**9'
        }
        for unicode_char, ascii_notation in unicode_superscripts.items():
            barrier_str = barrier_str.replace(unicode_char, ascii_notation)
            
        # LaTeX escape characters
        barrier_str = barrier_str.replace('\\\\', '\\').replace('\\ ', ' ')
        
        # LaTeX multiplication
        barrier_str = barrier_str.replace('\\cdot', '*')
        barrier_str = barrier_str.replace('\\times', '*')
        barrier_str = barrier_str.replace('·', '*')  
        
        # subscript notation x_i -> xi
        barrier_str = re.sub(r'x_(\d+)', r'x\1', barrier_str)
        
        barrier_str = re.sub(r'x(\d+)\^(\d+)', r'x\1**\2', barrier_str)                     # x1^4 -> x1**4
        barrier_str = re.sub(r'x\^(\d+)', r'x**\1', barrier_str)                            # x^4 -> x**4
        
        # general superscript with parentheses like (--)^n -> (--)**n
        barrier_str = re.sub(r'\^(\d+)', r'**\1', barrier_str)
        
        # standalone ^ to **
        barrier_str = barrier_str.replace('^', '**')
        
        # backslashes
        barrier_str = re.sub(r'\\', '', barrier_str)
        
        # Clean formatting
        barrier_str = barrier_str.strip().strip('"\'.,;:')
        
        barrier_str = re.sub(r'(\d)([x])', r'\1*\2', barrier_str)                           # 2x -> 2*x
        barrier_str = re.sub(r'(x\d+)([x])', r'\1*\2', barrier_str)                         # x1x2 -> x1*x2
        barrier_str = re.sub(r'([x\d])\((?!.*(?:sin|cos))', r'\1*(', barrier_str)           # x( -> x*( 
        barrier_str = re.sub(r'\)([x\d])', r')*\1', barrier_str)                            # )x -> )*x
        
        # common issues
        barrier_str = re.sub(r'\s+', ' ', barrier_str)
        
        # balance parentheses
        barrier_str = balance_parentheses(barrier_str)
        
        # trailing operators
        barrier_str = re.sub(r'[+\-\*]+\s*$', '', barrier_str).strip()
        
        # extract mathematical expression
        extracted_expr = extract_mathematical_expression(barrier_str)
        
        if extracted_expr:
            barrier_str = extracted_expr
        
        # cleanup
        barrier_str = barrier_str.strip()
        
        # x1-x10 not x₁-x₁₀
        for i in range(1, 11):
            barrier_str = re.sub(f'x₍{i}₎', f'x{i}', barrier_str)
        
        # one more parentheses balance check
        barrier_str = balance_parentheses(barrier_str)
        
        # check basic structure including variables x1-x10
        if len(barrier_str) < 3 or not (re.search(r'x\d+', barrier_str) or re.search(r'(?:sin|cos)\(', barrier_str)):
            return ""
        
        # ensure it doesn't end with operators
        barrier_str = re.sub(r'[+\-\*]+$', '', barrier_str).strip()
        
        if barrier_str:
            logger.info(f"Cleaned barrier expression: '{barrier_str}'")
        
        return barrier_str
        
    except Exception as e:
        logger.warning(f"Error cleaning barrier '{barrier_str}': {e}")
        return ""

def balance_parentheses(expression: str) -> str:
    # balance parentheses

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

def extract_mathematical_expression(text: str) -> str:
    # extract mathematical expression
    try:
        math_patterns = [
            r'((?:[-+]?\s*\d*\.?\d*\s*\*?\s*(?:x\d+(?:\*\*\d+)?(?:\s*\*\s*x\d+(?:\*\*\d+)?)*|(?:sin|cos)\([^)]+\))(?:\s*[+\-]\s*\d*\.?\d*\s*\*?\s*(?:x\d+(?:\*\*\d+)?(?:\s*\*\s*x\d+(?:\*\*\d+)?)*|(?:sin|cos)\([^)]+\)))*(?:\s*[+\-]\s*\d+\.?\d*)?))',           
            r'((?:[-+]?\s*\d*\.?\d*\s*\*?\s*x\d+(?:\*\*\d+)?(?:\s*\*\s*x\d+(?:\*\*\d+)?)*(?:\s*[+\-]\s*\d*\.?\d*\s*\*?\s*x\d+(?:\*\*\d+)?(?:\s*\*\s*x\d+(?:\*\*\d+)?)*)*(?:\s*[+\-]\s*\d+\.?\d*)?))',
            r'((?:[-+]?\d*\.?\d*\*?(?:sin|cos)\([^)]+\)|[-+]?\d*\.?\d*\*?x\d+(?:\*\*\d+)?(?:\*x\d+(?:\*\*\d+)?)*|[-+]?\d*\.?\d+)(?:\s*[+\-]\s*(?:[-+]?\d*\.?\d*\*?(?:sin|cos)\([^)]+\)|[-+]?\d*\.?\d*\*?x\d+(?:\*\*\d+)?(?:\*x\d+(?:\*\*\d+)?)*|[-+]?\d*\.?\d+))*)',
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
                
                if (len(extracted) > 3 and 
                    (re.search(r'x\d+', extracted) or re.search(r'(?:sin|cos)\(', extracted)) and
                    len(extracted) > best_length):
                    
                    balanced_extracted = balance_parentheses(extracted)
                    
                    if validate_trigonometric_syntax(balanced_extracted):
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

def validate_trigonometric_syntax(expression: str) -> bool:
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
                # must contain x1-x10 or mathematical expressions
                if not (re.search(r'x\d+', inner) or re.search(r'[+\-\*\d]', inner)):
                    logger.warning(f"Invalid trigonometric function content: {func}")
                    return False
            else:
                logger.warning(f"Empty trigonometric function: {func}")
                return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error validating trigonometric syntax in '{expression}': {e}") 
        return False

def parse_barrier_certificate(barrier_str: str) -> Tuple[Optional[sp.Expr], List[sp.Symbol]]:
    # parse barrier certificate string into symbolic expression
    try:
        cleaned_str = clean_and_extract_barrier(barrier_str)
        if not cleaned_str:
            logger.warning("Empty barrier string after cleaning")
            return None, []
        
        logger.info(f"Parsing cleaned barrier: '{cleaned_str}'")
        
        # extract variables
        variables = sorted(set(re.findall(r'\bx\d+\b', cleaned_str)), key=lambda x: int(x[1:]))
        if not variables:
            trig_vars = re.findall(r'(?:sin|cos)\([^)]*\b(x\d+)\b[^)]*\)', cleaned_str)
            if trig_vars:
                variables = sorted(set(trig_vars), key=lambda x: int(x[1:]))
            else:
                logger.error(f"No variables found in barrier expression")
                return None, []
        
        var_symbols = [sp.Symbol(var, real=True) for var in variables]
        
        try:
            expr_str = cleaned_str
            
            expr_str = re.sub(r'(\d)([x])', r'\1*\2', expr_str)             # 2x -> 2*x
            expr_str = re.sub(r'([x\d])\(', r'\1*(', expr_str)              # x( -> x*(
            expr_str = re.sub(r'\)([x\d])', r')*\1', expr_str)              # )x -> )*x
            
            expr_str = re.sub(r'sin\*\(', 'sin(', expr_str)
            expr_str = re.sub(r'cos\*\(', 'cos(', expr_str)
            
            expr_str = re.sub(r'(\d)\.(\d)', r'\1.\2', expr_str)
            
            logger.info(f"Final expression for parsing: '{expr_str}'")
            
            expr = sp.sympify(expr_str, evaluate=True)
            
            # validate
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
