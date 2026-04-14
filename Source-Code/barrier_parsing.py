import sympy as sp
from typing import List, Tuple, Optional
import re
import logging

logger = logging.getLogger(__name__)


def clean_barrier_expression(synthesized_barrier):
    if not synthesized_barrier or not isinstance(synthesized_barrier, str):
        return ""

    synthesized_barrier = re.sub(r'B\(x\)\s*=\s*', '', synthesized_barrier, flags=re.IGNORECASE)
    synthesized_barrier = re.sub(r'barrier\s*certificate\s*:?\s*', '', synthesized_barrier, flags=re.IGNORECASE)

    unicode_subscripts = {'₀':'0','₁':'1','₂':'2','₃':'3','₄':'4','₅':'5','₆':'6','₇':'7','₈':'8','₉':'9'}
    for u, a in unicode_subscripts.items():
        synthesized_barrier = synthesized_barrier.replace(u, a)

    unicode_superscripts = {'²':'**2','³':'**3','⁴':'**4','⁵':'**5','⁶':'**6','⁷':'**7','⁸':'**8','⁹':'**9'}
    for u, a in unicode_superscripts.items():
        synthesized_barrier = synthesized_barrier.replace(u, a)

    synthesized_barrier = synthesized_barrier.replace('\\\\', '\\').replace('\\ ', ' ')
    synthesized_barrier = synthesized_barrier.replace('\\cdot', '*').replace('\\times', '*').replace('·', '*')

    synthesized_barrier = re.sub(r'x_(\d+)', r'x\1', synthesized_barrier)
    synthesized_barrier = re.sub(r'x(\d+)\^(\d+)', r'x\1**\2', synthesized_barrier)
    synthesized_barrier = re.sub(r'x\^(\d+)', r'x**\1', synthesized_barrier)
    synthesized_barrier = re.sub(r'\^(\d+)', r'**\1', synthesized_barrier)
    synthesized_barrier = synthesized_barrier.replace('^', '**')
    synthesized_barrier = re.sub(r'\\', '', synthesized_barrier)

    synthesized_barrier = synthesized_barrier.strip().strip('"\'.,;:')
    synthesized_barrier = re.sub(r'(\d)([x])', r'\1*\2', synthesized_barrier)
    synthesized_barrier = re.sub(r'(x\d+)([x])', r'\1*\2', synthesized_barrier)
    synthesized_barrier = re.sub(r'([x\d])\((?!.*(?:sin|cos))', r'\1*(', synthesized_barrier)
    synthesized_barrier = re.sub(r'\)([x\d])', r')*\1', synthesized_barrier)
    synthesized_barrier = re.sub(r'\s+', ' ', synthesized_barrier)

    synthesized_barrier = balance_parentheses(synthesized_barrier)
    synthesized_barrier = re.sub(r'[+\-\*]+\s*$', '', synthesized_barrier).strip()

    extracted = extract_mathematical_expression(synthesized_barrier)
    if extracted:
        synthesized_barrier = extracted

    synthesized_barrier = balance_parentheses(synthesized_barrier.strip())

    if len(synthesized_barrier) < 3 or not (re.search(r'x\d+', synthesized_barrier) or re.search(r'(?:sin|cos)\(', synthesized_barrier)):
        return ""

    synthesized_barrier = re.sub(r'[+\-\*]+$', '', synthesized_barrier).strip()
    return synthesized_barrier


def balance_parentheses(expression):
    open_p = expression.count('(')
    close_p = expression.count(')')

    if open_p > close_p:
        expression += ')' * (open_p - close_p)
    elif close_p > open_p:
        extra = close_p - open_p
        for _ in range(extra):
            idx = expression.rfind(')')
            if idx != -1:
                expression = expression[:idx] + expression[idx+1:]

    return expression


def extract_mathematical_expression(text):
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
            if (len(extracted) > 3
                    and (re.search(r'x\d+', extracted) or re.search(r'(?:sin|cos)\(', extracted))
                    and len(extracted) > best_length):
                balanced = balance_parentheses(extracted)
                if validate_trigonometric_syntax(balanced):
                    best_match = balanced
                    best_length = len(balanced)

    return best_match if best_length > 10 else text


def validate_trigonometric_syntax(barrier_expression):
    for fn in re.findall(r'(?:sin|cos)\([^)]*\)', barrier_expression):
        if fn.count('(') != fn.count(')'):
            return False
        inner = re.search(r'(?:sin|cos)\(([^)]*)\)', fn)
        if not inner or not inner.group(1).strip():
            return False
        if not (re.search(r'x\d+', inner.group(1)) or re.search(r'[+\-\*\d]', inner.group(1))):
            return False
    return True


def parse_barrier_certificate(synthesized_barrier):
    cleaned = clean_barrier_expression(synthesized_barrier)
    if not cleaned:
        logger.error("Empty barrier string after cleaning")
        return None, []


    barrier_variables = sorted(set(re.findall(r'\bx\d+\b', cleaned)), key=lambda x: int(x[1:]))
    if not barrier_variables:
        trig_vars = re.findall(r'(?:sin|cos)\([^)]*\b(x\d+)\b[^)]*\)', cleaned)
        if trig_vars:
            barrier_variables = sorted(set(trig_vars), key=lambda x: int(x[1:]))
        else:
            logger.error("No variables found in barrier expression")
            return None, []

    var_symbols = [sp.Symbol(v, real=True) for v in barrier_variables]

    expr_str = cleaned
    expr_str = re.sub(r'(\d)([x])', r'\1*\2', expr_str)
    expr_str = re.sub(r'([x\d])\(', r'\1*(', expr_str)
    expr_str = re.sub(r'\)([x\d])', r')*\1', expr_str)
    expr_str = re.sub(r'sin\*\(', 'sin(', expr_str)
    expr_str = re.sub(r'cos\*\(', 'cos(', expr_str)


    sympy_expr = sp.sympify(expr_str, evaluate=True)

    if sympy_expr.free_symbols or sympy_expr.has(sp.sin) or sympy_expr.has(sp.cos):
        return sympy_expr, var_symbols

    logger.warning("Expression has no variables or functions")
    return None, var_symbols