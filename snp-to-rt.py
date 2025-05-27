
from unified_planning.model import Problem, Fluent, Action, Expression


def snp_to_rt(snp_formulas):
    """
    Convert extracted SNP formulas into RT-compatible representations.

    Args:
        snp_formulas (List[Expression]): List of formulas to convert.
    """
    pass


class SNP_RT_Transformer:
    """
    Transforms a Simple Numeric Planning (SNP) problem into a Restricted Task (RT) problem,
    where all preconditions must be of the form x_i > c_i.
    """

    def __init__(self):
        """
        Initialize transformer state.
        """
        self.problem = None
        self.substitute_dict = {}

    def transform(self, problem: Problem) -> Problem:
        """
        Main entry point: transforms the given SNP problem into an RT-compliant version.

        Args:
            problem (Problem): The input SNP planning problem.

        Returns:
            Problem: A new RT-compliant planning problem.
        """
        pass

    def extract_snp_formulas(self):
        """
        Walk through the problem and identify all preconditions that are not RT-compatible.
        """
        pass

    def clone_prob(self) -> Problem:
        """
        Clone the original problem and substitute all SNP constructs with RT-compatible ones.

        Returns:
            Problem: A fully transformed RT version of the original problem.
        """
        pass

    def is_rt_condition(self, expr: Expression) -> bool:
        """
        Check whether an expression is a valid RT precondition (i.e., x > c).

        Args:
            expr (Expression): The expression to evaluate.

        Returns:
            bool: True if expression is RT-compliant, False otherwise.
        """
        pass

    def create_aux_fluent(self, expr: Expression) -> Fluent:
        """
        Create an auxiliary fluent to replace a complex SNP numeric expression.

        Args:
            expr (Expression): The expression to abstract.

        Returns:
            Fluent: A new fluent representing the expression's value.
        """
        pass

    def substitute_expression(self, expr: Expression) -> Expression:
        """
        Substitute a formula using the substitute_dict.

        Args:
            expr (Expression): Original expression.

        Returns:
            Expression: Transformed expression.
        """
        pass

    def substitute_preconditions(self, action: Action) -> None:
        """
        Apply substitutions to an action's preconditions.
        """
        pass

    def substitute_effects(self, action: Action) -> None:
        """
        Apply substitutions to an action's effects if necessary.
        """
        pass
