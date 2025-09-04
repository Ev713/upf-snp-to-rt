from collections import defaultdict

import numpy as np
from unified_planning.model.multi_agent import *
from unified_planning.engines.mixins import CompilerMixin, CompilationKind
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPUsageError
from typing import Optional, List, OrderedDict
from unified_planning.shortcuts import *
import pandas as pd


# CONST = CONSTANTS[2]


def is_constant(node):
    return node.node_type in [OperatorKind.REAL_CONSTANT, OperatorKind.INT_CONSTANT]


def simple_fluent(node):
    return node.node_type in [OperatorKind.FLUENT_EXP]


def get_operator_as_function(op):
    return {OperatorKind.PLUS: Plus,
            OperatorKind.EQUALS: Equals,
            OperatorKind.MINUS: Minus,
            OperatorKind.TIMES: Times,
            OperatorKind.LT: LT,
            OperatorKind.LE: LE,
            }[op]

def get_fluent_from_simple_fluent(fluent):
    return str(fluent).split('(')[0]

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
    coeffs = {1: 0}
    while stack:
        node, coeff = stack.pop()

        if is_constant(node):
            coeffs[1] += coeff * constant_value(node)

        elif simple_fluent(node):
            params = node.args
            fluent = get_fluent_from_simple_fluent(node)
            fluent_repr = (fluent, params)
            coeffs[fluent_repr] = coeffs.get(fluent_repr, 0) + coeff

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

    return coeffs


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
                self.fluent_map[agent.name][fluent.name] = {}
                self.fluent_map[agent.name][fluent.name]['type'] = fluent.type
                self.fluent_map[agent.name][fluent.name]['args'] = fluent.signature
        return self.fluent_map

    def createActionMap(self):
        for agent in self.problem.agents:
            self.action_map[agent.name] = OrderedDict()
            for action in agent.actions:
                self.action_map[agent.name][action.name] = OrderedDict()
                self.action_map[agent.name][action.name]['parameters']=action.parameters
                self.action_map[agent.name][action.name]['precs'] = action.preconditions
                self.action_map[agent.name][action.name]['effs'] = action.effects

    def createEmptyActionMap(self):
        empty_action_map = OrderedDict()
        for agent in self.problem.agents:
            empty_action_map[agent.name] = OrderedDict()
            for action in agent.actions:
                empty_action_map[agent.name][action.name] = OrderedDict()
                empty_action_map[agent.name][action.name]['parameters'] = []
                empty_action_map[agent.name][action.name]['precs'] = []
                empty_action_map[agent.name][action.name]['effs'] = []
        return empty_action_map

    def find_fluent(self, agent, fluent):
        name = str(fluent).split('(')[0]
        if name in self.fluent_map[agent]:
            return self.fluent_map[agent][name]
        elif name in self.fluent_map['env']:
            return self.fluent_map[agent][name]
        raise Exception(f"Fluent {name} not found")

    def get_new_lin_fluent(self, lin, agent, params):
        tot_lower_bound = 0
        tot_upper_bound = 0
        name = '_'
        for i, p in enumerate(params):
            f = self.find_fluent(agent, p)
            f_type = f['type']
            f_range = f_type.lower_bound, f_type.upper_bound
            coeff = lin[p]
            tot_lower_bound += min(coeff * f_range[0], coeff * f_range[1])
            tot_upper_bound += max(coeff * f_range[0], coeff * f_range[1])
            if coeff < 0:
                name += '-'
            elif len(name) > 1:
                name += '+'
            if abs(coeff) != 1:
                name += str(abs(coeff))
            name += str(p)
        return Fluent(name, RealType(tot_lower_bound, tot_upper_bound))


def fluent_column_name(fluent, param_idx):
    """Return a canonical column name for a fluent occurrence."""
    if fluent == 1:
        return "1"  # constant term, no index needed
    if isinstance(fluent, tuple):
        # fluent with parameters
        return f"{fluent[0]}({','.join('p'+str(i+1) for i in range(len(fluent[1])))})" + ("" if param_idx == 0 else f"_{param_idx+1}")
    else:
        # fluent with no parameters
        return fluent + ("" if param_idx == 0 else f"_{param_idx+1}")

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

    def create_readable_action_map(self):
        readable_action_map = self.swapper.createEmptyActionMap()
        largest_precondition_arity = 0
        for agent in self.swapper.action_map:
            for action in self.swapper.action_map[agent]:
                readable_action_map[agent][action]['precs'] = {'lin': [], 'non_lin': []}
                for prec in self.swapper.action_map[agent][action]['precs']:
                    if not prec_is_comparison(prec):
                        readable_action_map[agent][action]['precs']['non_lin'].append(prec)
                        continue
                    lin_node1, lin_node2 = prec.args
                    lin1 = parse_lin(lin_node1)
                    lin2 = parse_lin(lin_node2)
                    prec_simple_fluents = set(lin1.keys()) | set(lin2.keys())
                    operator = prec.node_type
                    prec_fluent = {}
                    for p in prec_simple_fluents:
                        if p != 1:
                            prec_fluent[p] = lin1.get(p, 0) - lin2.get(p, 0)
                    value = lin2.get(1, 0) - lin1.get(1, 0)

                    largest_precondition_arity = max(largest_precondition_arity, len(prec_fluent))
                    readable_action_map[agent][action]['precs']['lin'].append((prec_fluent, operator, value))
                readable_action_map[agent][action]['effs'] = {'lin': [], 'non_lin': []}

                for eff in self.swapper.action_map[agent][action]['effs']:
                    target_var, expr_node = eff.fluent, eff.value
                    # Check if the effect is linear
                    try:
                        lin_expr = parse_lin(expr_node)  # rhs = right-hand side of assignment
                    except Exception:  # or your parse_lin failure indicator
                        readable_action_map[agent][action]['effs']['non_lin'].append(eff)
                        continue
                    # Decompose into a mapping of fluent → coefficient
                    effect_fluent = {}
                    for p in lin_expr:
                        effect_fluent[p] = lin_expr[p]

                    # Save: (target variable, linear assignment dict)
                    readable_action_map[agent][action]['effs']['lin'].append((target_var, effect_fluent))
        return readable_action_map

    def is_private(self, fluent, agent):
        return fluent in self.swapper.fluent_map[agent]


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

        readable_action_map = self.create_readable_action_map()
        #print(readable_action_map)
        fluents = set()
        for agent, actions in readable_action_map.items():
            for action, data in actions.items():
                for lin_prec, _, _ in data['precs']['lin']:
                    for f_args in lin_prec:
                        fluents.add(f_args[0])

        # Make a sorted list to fix column order
        fluents = sorted(fluents, key=lambda x: str(x))
        print(fluents)
        column_counter = defaultdict(int)
        columns = ['agent', 'action', 'precondition_index', 'operator', 'value']

        for agent, actions in readable_action_map.items():
            for action, data in actions.items():
                for lin_prec, _, _ in data['precs']['lin']:
                    # Count occurrences of each fluent in this precondition
                    local_counter = defaultdict(int)
                    for f_args in lin_prec:
                        f = f_args[0]
                        count = local_counter[f]
                        col_name = fluent_column_name(f, count)
                        local_counter[f] += 1
                        if col_name not in column_counter:
                            column_counter[col_name] = None  # add new column
        columns += sorted(column_counter.keys())  # alphabetical order

        # 2️⃣ Create empty DataFrame
        df_precs = pd.DataFrame(columns=columns)

        # 3️⃣ Fill rows
        rows = []
        for agent, actions in readable_action_map.items():
            for action, data in actions.items():
                print(action)
                pass
                for idx, (lin_prec, operator, value) in enumerate(data['precs']['lin']):
                    row = {'agent': agent, 'action': action, 'precondition_index': idx,
                           'operator': operator, 'value': value}
                    # Track local occurrences
                    local_counter = defaultdict(int)
                    for f_args in lin_prec:
                        f = f_args[0]
                        count = local_counter[f]
                        col_name = fluent_column_name(f, count)
                        row[col_name] = lin_prec[f_args]
                        local_counter[f] += 1
                    # Fill remaining columns with 0
                    for col in df_precs.columns:
                        if col not in row:
                            row[col] = 0
                    rows.append(row)

        df_precs = pd.concat([df_precs, pd.DataFrame(rows)], ignore_index=True)

        pd.set_option('display.max_columns', None)

        # Show all rows
        pd.set_option('display.max_rows', None)

        # Optionally, increase column width so contents are not truncated
        pd.set_option('display.max_colwidth', None)

        # Now printing the DataFrame shows full content
        print(df_precs)

        # new fluent - defined by coefficients, fluents, object types of fluent params
        # precondition, effect - defined by new fluent with action parameter objects as parameters
        # V - Step 1: Write every action and precondition and effect in a readable form.
        # Step 1.5 represent each new fluent as vector on table: f1_p1, f1_p2, ... f2_p1, ... where number of each
        # parameters for each fluent is determined by the largest number of same fluents in a precondition.
        # Step 2: Use preconditions to write out which new fluents will be created.
        # Step 3: Prune duplicates.
        # Step 3.5: Create new fluents
        # Step 4: create new preconditions and effects for every action



        breakpoint()
        new_fluents_table = pd.DataFrame()



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
