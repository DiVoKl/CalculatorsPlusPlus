from abc import ABC, abstractmethod
import sympy as sp
from typing import Dict, Any
from utils import parse_expression, format_result

class BaseCalculator(ABC):
    """
    Abstract Base Class that handles symbol initialization, expression storage,
    and safe execution wrappers for all calculus modules.
    """
    def __init__(self, expr_str: str, symbols_str: str = "x y z t"):
        self.raw_expr = expr_str
        self.expr, self.symbols = parse_expression(expr_str, symbols_str)
        self.x, self.y, self.z, self.t = self.symbols[:4]

    @abstractmethod
    def calculate(self, **kwargs) -> Dict[str, Any]:
        """Abstract execution method enforced for all extending modules."""
        pass

    def safe_execute(self, func, *args, **kwargs) -> Dict[str, Any]:
        """
        Executes a SymPy function safely, isolating computational runtime exceptions
        and formatting successful outputs.
        """
        try:
            res = func(*args, **kwargs)
            if isinstance(res, sp.Expr):
                return format_result(res)
            return {"result": res, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}