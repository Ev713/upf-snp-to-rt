import numpy as np
import unified_planning as up
import unified_planning.model.operators
from unified_planning.model import Problem, Fluent, Action, Expression, ProblemKind
from unified_planning.model.multi_agent import *
from unified_planning.model.operators import CONSTANTS
from unified_planning.model.operators import OperatorKind as OPS
from unified_planning.engines import Engine
from unified_planning.engines.mixins import CompilerMixin, CompilationKind
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPUsageError
from typing import Optional, List, OrderedDict
from enum import Enum, auto
from collections import defaultdict
from unified_planning.model import *
from unified_planning.shortcuts import *
from typing import cast


# CONST = CONSTANTS[2]


def is_constant(node):
    return node.node_type in [OperatorKind.REAL_CONSTANT, OperatorKind.INT_CONSTANT]


def is_parameter(node):
    return node.node_type in [OperatorKind.FLUENT_EXP]


def get_operator_as_function(op):
    return {OperatorKind.PLUS: Plus,
            OperatorKind.MINUS: Minus,
            OperatorKind.TIMES: Times,
            OperatorKind.LT: LT,
            OperatorKind.LE: LE,
            }[op]


def is_operator(node):
    return node.node_type in [OperatorKind.PLUS, OperatorKind.MINUS, OperatorKind.TIMES]


def prec_is_comparison(node):
    return node.node_type in [OperatorKind.LE, OperatorKind.EQUALS, OperatorKind.LT]


def constant_value(const):
    v = const.type.lower_bound
    if v == const.type.upper_bound:
        return v
    raise Exception('Couldn\'t parse constant')

def get_actions(problem):
    if isinstance(problem, MultiAgentProblem):
        actions = []
        for agent in problem.agents:
            actions += agent.actions
        return actions
    return problem.actions

def parse_lin(lin):
    stack = [(lin, 1)]  # Each item is (sub_expr, multiplier)
    parameters = set()
    result = {}

    while stack:
        node, coeff = stack.pop()

        if is_constant(node):
            result[1] = result.get(1, 0) + coeff * constant_value(node)

        elif is_parameter(node):
            parameters.add(node)
            result[node] = result.get(node, 0) + coeff

        elif is_operator(node):

            a, b = node.args
            if node.node_type in [OperatorKind.PLUS, OperatorKind.MINUS]:
                if node.node_type == OperatorKind.MINUS:
                    coeff *= -1
                stack.append((a, coeff))
                stack.append((b, coeff))

            elif node.node_type == OperatorKind.TIMES:
                if is_constant(a):
                    stack.append((b, coeff * constant_value(a)))
                elif is_constant(b):
                    stack.append((a, coeff * constant_value(b)))
                else:
                    raise ValueError("Non-linear multiply detected: both operands are non-constants")
            else:
                raise ValueError(f"Unknown operator")
        else:
            raise ValueError(f"Invalid expression node: {node}")

    return result, parameters


class ProblemSwapper:
    def __init__(self, problem: Problem):
        self.problem = problem
        self.objects = []
        self.fluent_map = OrderedDict()

    def get_objects(self):
        self.objects = self.problem.all_objects
        return self.objects

    def createEmptyProblem(self) -> Problem:
        pass

    def createFluentMap(self):
        pass

    def createActionMap(self):
        pass


class SAProblemSwapper(ProblemSwapper):
    def createEmptyProblem(self):
        return Problem()

    def createFluentMap(self):
        pass

    def createActionMap(self):
        pass

class MAProblemSwapper(ProblemSwapper):
    problem: MultiAgentProblem  # class-level override

    def __init__(self, problem: MultiAgentProblem):
        super().__init__(problem)
        self.action_map = OrderedDict()

    def createEmptyProblem(self):
        return MultiAgentProblem()

    def createFluentMap(self):
        self.fluent_map['env'] = OrderedDict()
        for fluent in self.problem.ma_environment.fluents:
            self.fluent_map['env'][fluent.name] = fluent
        for agent in self.problem.agents:
            self.fluent_map[agent.name] = OrderedDict()
            for fluent in agent.fluents:
                self.fluent_map[agent.name][fluent.name] = fluent
        return self.fluent_map

    def createActionMap(self):
        for agent in self.problem.agents:
            self.action_map[agent.name] = OrderedDict()
            for action in agent.actions:
                self.action_map[agent.name][action.name] = OrderedDict()
                self.action_map[agent.name][action.name]['parameters']=action.parameters
                self.action_map[agent.name][action.name]['precs'] = action.preconditions
                self.action_map[agent.name][action.name]['effs'] = action.effects



class SNP_RT_Transformer(CompilerMixin, Engine):
    """
    Transforms a Simple Numeric Planning (SNP) problem into a Restricted Task (RT) problem,
    where all preconditions must be of the form x_i > c_i.
    """

    SNP_TO_RT = object()

    @property
    def name(self) -> str:
        return "SNP_RT_Transformer"

    def supported_kind(self) -> ProblemKind:
        # Return the most general kind that this transformer can support
        # You can customize this further based on your transformation rules
        return ProblemKind.NUMERIC_STATE_FLUENTS | ProblemKind.ACTION_BASED

    def supports(self, problem_kind: ProblemKind) -> bool:
        return problem_kind <= self.supported_kind()

    def __init__(self):
        Engine.__init__(self)
        CompilerMixin.__init__(self, default=SNP_RT_Transformer.SNP_TO_RT)
        self.problem = None
        self.substitute_dict = {}

    def transform(self, problem: Problem) -> Problem:
        """Transforms the given SNP problem into an RT-compliant version."""
        self.problem = problem
        if isinstance(problem, MultiAgentProblem):
            self.swapper = MAProblemSwapper(problem)
        else:
            self.swapper = SAProblemSwapper(problem)
        rt_prob=self.swapper.createEmptyProblem()
        for o in self.swapper.get_objects():
            rt_prob.add_object(o.name, o.type)
        self.swapper.createFluentMap()
        self.swapper.createActionMap()


        params, formulas, values, operators = self.extract_formulas()
        rt_params, rt_vals, operators = self.create_rt_formulas(params, formulas, values, operators)
        # TODO: In this line, need to create new problem and substitute params, then use:
        rt_fluents = self.make_formulas_into_fluents(params, rt_params, rt_vals, operators)
        # TODO: and add the corresponding precons and effects into the new problem
        return self.clone_prob(rt_fluents)


    def create_rt_formulas(self, params, formulas, values, operators):
        param_ids = {par: i for i, par in enumerate(params)}
        rt_forms = []
        for dict_form in formulas:
            f = np.zeros(len(params))
            for p in dict_form:
                f[param_ids[p]] = dict_form[p]
            rt_forms.append(f)

        modified_values = values.copy()
        rt_params = []
        for i, row in enumerate(rt_forms):
            nonzero_indices = np.flatnonzero(row)
            if nonzero_indices.size > 0:
                first_nonzero = row[nonzero_indices[0]]
                row /= first_nonzero
                modified_values[i] /= first_nonzero
            rt_param = tuple(row)
            rt_params.append(rt_param)

        return rt_params, modified_values, operators

    def make_formulas_into_fluents(self, params, rt_formulas, values, operators):
        rt_fluents = []
        for i, rt_form in enumerate(rt_formulas):
            non_zeros = list(np.flatnonzero(rt_form))
            if len(non_zeros) == 0:
                rt_fluents.append(get_operator_as_function(operators[i])(0, values[i]))
            new_fluent = params[non_zeros.pop(0)]
            while len(non_zeros) > 0:
                k = non_zeros.pop(0)
                new_fluent = Plus(new_fluent, Times(rt_form[k], params[k]))
            rt_fluents.append(get_operator_as_function(operators[i])(new_fluent, values[i]))
        return rt_fluents

    def extract_formulas(self):
        """Extract the linear formulas from the preconditions and effects of actions."""
        params = set()
        operators = []
        formulas = []
        values = []

        actions = get_actions(self.problem)
        for action in actions:
            for prec in action.preconditions:
                if not prec_is_comparison(prec):
                    continue
                lin_node1, lin_node2 = prec.args
                lin1, params1 = parse_lin(lin_node1)
                lin2, params2 = parse_lin(lin_node2)
                prec_params = params1 | params2
                operator = prec.node_type
                new_prec_left = {}
                for p in prec_params:
                    new_prec_left[p] = lin1.get(p, 0) - lin2.get(p, 0)

                params = params | prec_params
                formulas.append(new_prec_left)
                values.append(lin2.get(1, 0) - lin1.get(1, 0))
                operators.append(operator)

        return sorted(list(params)), formulas, values, operators

    def clone_prob(self, rt_fluents) -> Problem:
        raise NotImplementedError

    def is_rt_condition(self, expr: Expression) -> bool:
        """Check if expression is RT-compatible (i.e., x > c)."""
        # TODO: Fix whatever is here
        # Dummy placeholder
        return True

    def substitute_expression(self, expr: Expression) -> Expression:
        """Substitute using the dictionary."""
        return self.substitute_dict.get(expr, expr)

    def substitute_preconditions(self, action: Action) -> None:
        """Substitute action preconditions."""
        pass

    def substitute_effects(self, action: Action) -> None:
        """Substitute action effects."""
        pass

    # --- CompilerMixin required methods ---

    @staticmethod
    def supports_compilation(compilation_kind: CompilationKind) -> bool:
        return compilation_kind == CompilationKind.USERTYPE_FLUENTS_REMOVING

    @staticmethod
    def resulting_problem_kind(
            problem_kind: ProblemKind, compilation_kind: Optional[CompilationKind] = None
    ) -> ProblemKind:
        return problem_kind  # Assuming kind remains valid after RT transformation

    def _compile(
            self, problem: Problem, compilation_kind: CompilationKind
    ) -> CompilerResult:
        if compilation_kind != CompilationKind.USERTYPE_FLUENTS_REMOVING:
            raise UPUsageError("SNP_RT_Transformer only supports USERTYPE_FLUENTS_REMOVING kind.")

        new_problem = self.transform(problem)

        def map_plan(plan):
            # If plan mapping is needed between old and new problems
            return plan

        return CompilerResult(
            problem=new_problem,
            plan_rewriter=map_plan,
            name="SNP_RT_Transformer",
            logs=["Transformed SNP to RT"]
        )
