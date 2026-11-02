from abc import ABC, abstractmethod
import sympy as sp
from typing import Dict, Any
from functools import wraps
from .utils import parse_expression, format_result

def safe_step(func):
    """
    Decorator that automatically wraps a method in a try-except block
    to remove repetitive error handling across all modules.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {"error": str(e), "success": False}
    return wrapper

class BaseCalculator(ABC):
    def __init__(self, expr_str: str, symbols_str: str = "x y z t"):
        self.raw_expr = expr_str
        self.expr, self.symbols = parse_expression(expr_str, symbols_str)
        self.x, self.y, self.z, self.t = self.symbols[:4]

    @abstractmethod
    def calculate(self, **kwargs) -> Dict[str, Any]:
        pass

    def safe_execute(self, func, *args, **kwargs) -> Dict[str, Any]:
        try:
            res = func(*args, **kwargs)
            if isinstance(res, sp.Expr):
                return format_result(res)
            return {"result": res, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    # --- NEW HELPER METHODS ---
    def parse_secondary(self, expr_str: str) -> sp.Expr:
        """Parses a secondary string (like a constraint) without redundant imports."""
        parsed, _ = parse_expression(expr_str, "x y z t")
        return parsed

    def get_gradient_2d(self):
        """Returns the first partial derivatives [fx, fy] to avoid repeating sp.diff."""
        return sp.diff(self.expr, self.x), sp.diff(self.expr, self.y)