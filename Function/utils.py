import sympy as sp
from typing import Dict, Any, Tuple

def parse_expression(expr_str: str, symbols_str: str = "x y z t") -> Tuple[sp.Expr, Tuple[sp.Symbol, ...]]:
    """
    Normalizes mathematical string syntax, instantiates multi-variable symbols,
    and returns a symbolic SymPy expression object.
    """
    try:
        symbols = sp.symbols(symbols_str)
        cleaned_str = expr_str.replace('^', '**')  # Converts user exponent syntax
        parsed_expr = sp.sympify(cleaned_str)
        return parsed_expr, symbols
    except Exception as e:
        raise ValueError(f"Expression parsing error for '{expr_str}': {str(e)}")

def format_result(result: sp.Expr) -> Dict[str, Any]:
    """
    Wraps raw SymPy evaluation outputs into a standardized dictionary containing
    plain text, LaTeX representations, and raw symbolic objects.
    """
    return {
        "text": str(result),
        "latex": sp.latex(result),
        "sympy_obj": result,
        "success": True
    }