import sympy as sp
from typing import List, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)


def clean_barrier_expression(synthesized_barrier: str) -> str:
    # extract and clean barrier certificate 

    if not synthesized_barrier or not isinstance(synthesized_barrier, str):
        return ""
    
    try:
        # common prefixes
        synthesized_barrier = re.sub(r'B\(x\)\s*=\s*', '', synthesized_barrier, flags=re.IGNORECASE)
        synthesized_barrier = re.sub(r'barrier\s*certificate\s*:?\s*', '', synthesized_barrier, flags=re.IGNORECASE)
        
        # unicode subscripts handling
        unicode_subscripts = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
        }
        for unicode_char, ascii_char in unicode_subscripts.items():
            synthesized_barrier = synthesized_barrier.replace(unicode_char, ascii_char)
        
        # unicode superscripts handling 
        unicode_superscripts = {
            '²': '**2', '³': '**3', '⁴': '**4', '⁵': '**5',
            '⁶': '**6', '⁷': '**7', '⁸': '**8', '⁹': '**9'
        }
        for unicode_char, ascii_notation in unicode_superscripts.items():
            synthesized_barrier = synthesized_barrier.replace(unicode_char, ascii_notation)
            
        # LaTeX escape characters
        synthesized_barrier = synthesized_barrier.replace('\\\\', '\\').replace('\\ ', ' ')
        
        # LaTeX multiplication
        synthesized_barrier = synthesized_barrier.replace('\\cdot', '*')
        synthesized_barrier = synthesized_barrier.replace('\\times', '*')
        synthesized_barrier = synthesized_barrier.replace('·', '*')  
        
        # subscript notation x_i -> xi
        synthesized_barrier = re.sub(r'x_(\d+)', r'x\1', synthesized_barrier)
        
        synthesized_barrier = re.sub(r'x(\d+)\^(\d+)', r'x\1**\2', synthesized_barrier)                     # x1^4 -> x1**4
        synthesized_barrier = re.sub(r'x\^(\d+)', r'x**\1', synthesized_barrier)                            # x^4 -> x**4
        
        # general superscript with parentheses like (--)^n -> (--)**n
        synthesized_barrier = re.sub(r'\^(\d+)', r'**\1', synthesized_barrier)
        
        # standalone ^ to **
        synthesized_barrier = synthesized_barrier.replace('^', '**')
        
        # backslashes
        synthesized_barrier = re.sub(r'\\', '', synthesized_barrier)
        
        # Clean formatting
        synthesized_barrier = synthesized_barrier.strip().strip('"\'.,;:')
        
        synthesized_barrier = re.sub(r'(\d)([x])', r'\1*\2', synthesized_barrier)                           # 2x -> 2*x
        synthesized_barrier = re.sub(r'(x\d+)([x])', r'\1*\2', synthesized_barrier)                         # x1x2 -> x1*x2
        synthesized_barrier = re.sub(r'([x\d])\((?!.*(?:sin|cos))', r'\1*(', synthesized_barrier)           # x( -> x*( 
        synthesized_barrier = re.sub(r'\)([x\d])', r')*\1', synthesized_barrier)                            # )x -> )*x
        
        # common issues
        synthesized_barrier = re.sub(r'\s+', ' ', synthesized_barrier)
        
        # balance parentheses
        synthesized_barrier = balance_parentheses(synthesized_barrier)
        
        # trailing operators
        synthesized_barrier = re.sub(r'[+\-\*]+\s*$', '', synthesized_barrier).strip()
        
        # extract mathematical expression
        extracted_expr = extract_mathematical_expression(synthesized_barrier)
        
        if extracted_expr:
            synthesized_barrier = extracted_expr
        
        # cleanup
        synthesized_barrier = synthesized_barrier.strip()
        
        # x1-x10 not x₁-x₁₀
        for i in range(1, 11):
            synthesized_barrier = re.sub(f'x₍{i}₎', f'x{i}', synthesized_barrier)
        
        # one more parentheses balance check
        synthesized_barrier = balance_parentheses(synthesized_barrier)
        
        # check basic structure including variables x1-x10
        if len(synthesized_barrier) < 3 or not (re.search(r'x\d+', synthesized_barrier) or re.search(r'(?:sin|cos)\(', synthesized_barrier)):
            return ""
        
        # ensure it doesn't end with operators
        synthesized_barrier = re.sub(r'[+\-\*]+$', '', synthesized_barrier).strip()
        
        if synthesized_barrier:
            logger.info(f"Cleaned barrier expression: '{synthesized_barrier}'")

        return synthesized_barrier
        
    except Exception as e:
        logger.warning(f"Error cleaning barrier '{synthesized_barrier}': {e}")
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

def validate_trigonometric_syntax(barrier_expression: str) -> bool:
    try:
        trigonometric_functions = re.findall(r'(?:sin|cos)\([^)]*\)', barrier_expression)
        
        for function in trigonometric_functions:
            open_count = function.count('(')
            close_count = function.count(')')
            
            if open_count != close_count:
                logger.warning(f"Unbalanced parentheses in trigonometric function: {function}")
                return False
            
            inner_content = re.search(r'(?:sin|cos)\(([^)]*)\)', function)
            if inner_content and inner_content.group(1).strip():
                inner = inner_content.group(1)
                # must contain x1-x10 or mathematical expressions
                if not (re.search(r'x\d+', inner) or re.search(r'[+\-\*\d]', inner)):
                    logger.warning(f"Invalid trigonometric function content: {function}")
                    return False
            else:
                logger.warning(f"Empty trigonometric function: {function}")
                return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error validating trigonometric syntax in '{barrier_expression}': {e}") 
        return False

def parse_barrier_certificate(synthesized_barrier: str) -> Tuple[Optional[sp.Expr], List[sp.Symbol]]:
    # parse barrier certificate string into symbolic expression
    try:
        cleaned_barrier = clean_barrier_expression(synthesized_barrier)
        if not cleaned_barrier:
            print("ERROR: Empty barrier string after cleaning")
            return None, []
        
        logger.info(f"Parsing cleaned barrier: '{cleaned_barrier}'")
        
        # extract variables
        barrier_variables = sorted(set(re.findall(r'\bx\d+\b', cleaned_barrier)), key=lambda x: int(x[1:]))
        if not barrier_variables:
            trig_vars = re.findall(r'(?:sin|cos)\([^)]*\b(x\d+)\b[^)]*\)', cleaned_barrier)
            if trig_vars:
                barrier_variables = sorted(set(trig_vars), key=lambda x: int(x[1:]))
            else:
                print("ERROR: No variables found in barrier expression")
                return None, []
        
        var_symbols = [sp.Symbol(var, real=True) for var in barrier_variables]
        
        try:
            cleaned_barrier_expr = cleaned_barrier
            
            cleaned_barrier_expr = re.sub(r'(\d)([x])', r'\1*\2', cleaned_barrier_expr)             # 2x -> 2*x
            cleaned_barrier_expr = re.sub(r'([x\d])\(', r'\1*(', cleaned_barrier_expr)              # x( -> x*(
            cleaned_barrier_expr = re.sub(r'\)([x\d])', r')*\1', cleaned_barrier_expr)              # )x -> )*x
            
            cleaned_barrier_expr = re.sub(r'sin\*\(', 'sin(', cleaned_barrier_expr)
            cleaned_barrier_expr = re.sub(r'cos\*\(', 'cos(', cleaned_barrier_expr)
            
            cleaned_barrier_expr = re.sub(r'(\d)\.(\d)', r'\1.\2', cleaned_barrier_expr)
            
            logger.info(f"Final expression for parsing: '{cleaned_barrier_expr}'")
            
            cleaned_barrier_sympy = sp.sympify(cleaned_barrier_expr, evaluate=True)
            
            # validate
            if cleaned_barrier_sympy.free_symbols or (hasattr(cleaned_barrier_sympy, 'has') and (cleaned_barrier_sympy.has(sp.sin) or cleaned_barrier_sympy.has(sp.cos))):
                logger.info(f"Successfully parsed expression: {cleaned_barrier_sympy}")
                return cleaned_barrier_sympy, var_symbols
            else:
                logger.warning("Expression has no variables or functions")
                return None, var_symbols
                
        except sp.SympifyError as e:
            logger.warning(f"SymPy failed to parse expression '{cleaned_barrier}': {e}")
            return None, var_symbols
        except Exception as e:
            logger.warning(f"Failed to parse expression '{cleaned_barrier}': {e}")
            return None, var_symbols
        
    except Exception as e:
        logger.error(f"Error parsing barrier certificate '{synthesized_barrier}': {e}")
        return None, []