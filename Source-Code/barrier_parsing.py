import re
import logging
import sympy as sp

logger = logging.getLogger(__name__)

def clean_barrier_expression(synthesized_barrier):
    if not isinstance(synthesized_barrier, str):
        return ""
    
    synthesized_barrier = re.sub(r'(?i)^(B\(x\)\s*=\s*|barrier\s*certificate\s*:?\s*)', '', synthesized_barrier)
    
    subs = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    synthesized_barrier = synthesized_barrier.translate(subs)
    
    for u, a in [('²','**2'),('³','**3'),('⁴','**4'),('⁵','**5'),('⁶','**6'),('⁷','**7'),('⁸','**8'),('⁹','**9')]:
        synthesized_barrier = synthesized_barrier.replace(u, a)
    
    synthesized_barrier = re.sub(r'\\(?:cdot|times)|·', '*', synthesized_barrier)
    synthesized_barrier = synthesized_barrier.replace('\\\\', '\\').replace('\\ ', ' ')
    
    synthesized_barrier = re.sub(r'x_(\d+)', r'x\1', synthesized_barrier)
    synthesized_barrier = re.sub(r'x(\d+)\^(\d+)', r'x\1**\2', synthesized_barrier)
    synthesized_barrier = re.sub(r'([x)])\^(\d+)', r'\1**\2', synthesized_barrier)
    synthesized_barrier = synthesized_barrier.replace('^', '**')
    synthesized_barrier = re.sub(r'\\(?!\*\*)', '', synthesized_barrier)  
    
    synthesized_barrier = re.sub(r'(\d)(x\d+)', r'\1*\2', synthesized_barrier)
    synthesized_barrier = re.sub(r'(x\d+)(x\d+)', r'\1*\2', synthesized_barrier)
    synthesized_barrier = re.sub(r'([\d\)])(?=[(]?(?!sin|cos)[a-z])', r'\1*', synthesized_barrier)
    synthesized_barrier = re.sub(r'([a-z])(\d)', r'\1*\2', synthesized_barrier)
    
    synthesized_barrier = re.sub(r'\s+', ' ', synthesized_barrier).strip('"\'.,;:')
    
    open_p, close_p = synthesized_barrier.count('('), synthesized_barrier.count(')')
    if open_p > close_p:
        synthesized_barrier += ')' * (open_p - close_p)
    
    math_match = re.search(r'([-+]?\s*\d*\.?\d*\s*[*]?\s*(?:x\d+(?:\*\*\d+)?|(?:sin|cos)\([^)]+\))(?:\s*[+-]\s*(?:[-+]?\s*\d*\.?\d*\s*[*]?\s*(?:x\d+(?:\*\*\d+)?|(?:sin|cos)\([^)]+\)))*(?:\s*[+-]\s*\d*\.?\d*)?)', synthesized_barrier)
    if math_match and len(math_match.group(1)) > 10:
        synthesized_barrier = math_match.group(1)
    
    return synthesized_barrier if re.search(r'x\d+|(?:sin|cos)\(', synthesized_barrier) and len(synthesized_barrier) >= 3 else ""


def parse_barrier_certificate(synthesized_barrier):
    cleaned = clean_barrier_expression(synthesized_barrier)
    if not cleaned:
        logger.error("Empty barrier string after cleaning")
        return None, []
    
    barrier_variables = sorted(set(re.findall(r'\bx\d+\b', cleaned)), key=lambda x: int(x[1:]))
    if not barrier_variables:
        barrier_variables = sorted(set(re.findall(r'(?:sin|cos)\([^)]*\b(x\d+)\b', cleaned)), key=lambda x: int(x[1:]))
        if not barrier_variables:
            logger.error("No variables found in barrier expression")
            return None, []
    
    var_symbols = [sp.Symbol(v, real=True) for v in barrier_variables]
    
    cleaned = re.sub(r'(\d)(x\d+)', r'\1*\2', cleaned)
    cleaned = re.sub(r'(x\d+)(x\d+)', r'\1*\2', cleaned)
    cleaned = re.sub(r'sin\*\(|cos\*\(', r'sin(', cleaned)

    sympy_expr = sp.sympify(cleaned, evaluate=True)

    if sympy_expr.free_symbols or sympy_expr.has(sp.sin, sp.cos):
        return sympy_expr, var_symbols

    logger.warning("Expression has no variables or functions")
    return None, var_symbols