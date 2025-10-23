import itertools
from collections import defaultdict

import numpy as np
from unified_planning.model.multi_agent import *
from unified_planning.engines.mixins import CompilerMixin, CompilationKind
from unified_planning.engines.results import CompilerResult
from unified_planning.exceptions import UPUsageError
from typing import Optional, List, OrderedDict, Tuple
from unified_planning.shortcuts import *
import pandas as pd
from fractions import Fraction


# CONST = CONSTANTS[2]
ONE = ('1', ())
def fluent_column_name(fluent, param_idx):
    """Return a canonical column name for a fluent occurrence."""
    if fluent == 1:
        return "1"  # constant term, no index needed
    if isinstance(fluent, tuple):
        # fluent with parameters
        return f"{fluent[0]}({','.join('p'+str(i+1) for i in range(len(fluent[1])))})" + ("" if param_idx == 0 else f"#{param_idx+1}")
    else:
        # fluent with no parameters
        return fluent + ("" if param_idx == 0 else f"#{param_idx+1}")

def operator_to_symbol(operator):
    return {OperatorKind.LT:'<', OperatorKind.LE:'<=', OperatorKind.EQUALS: '='}[operator]

def normalize_dataframe(df, dont_change_columns=None, ignore_as_divisor_columns=None,  add_coeffs=True, operator_column=None):
    if dont_change_columns is None:
        dont_change_columns = []

    if ignore_as_divisor_columns is None:
        ignore_as_divisor_columns = []

    df_frac = df.copy()
    coeffs = []

    # Identify numeric columns only, excluding ignored ones
    numeric_cols = []
    for col in df_frac.columns:
        col_objects = list(df_frac[col])
        if all([isinstance(obj, int) or isinstance(obj, float) for obj in col_objects]):
            numeric_cols.append(col)

    # Convert all numeric entries in relevant columns to Fraction
    for col in numeric_cols:
        df_frac[col] = df_frac[col].apply(lambda x: Fraction(str(x)))

    # Normalize each row
    for i, row in df_frac.iterrows():
        divisor = None
        # find leftmost non-zero numeric value (excluding ignored)
        for col in numeric_cols:
            if row[col] != 0 and col not in ignore_as_divisor_columns:
                divisor = row[col]
                break
        if divisor is not None:
            for col in numeric_cols:
                if col not in dont_change_columns:
                    df_frac.at[i, col] = row[col] / divisor
        coeffs.append(divisor)
        if operator_column is not None and divisor < 0:
            df_frac.at[i, operator_column] = {'<=': '>=', '>=': '<=', '<': '>', '>': '<', '=': '='}[row[operator_column]]
    if add_coeffs:
        df_frac['coeff'] = coeffs

    return df_frac

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
            '>':GT,
            '>=':GE,
            '<': LT,
            '<=': LE,
            '=':Equals
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
    coeffs = {ONE: 0}
    while stack:
        node, coeff = stack.pop()

        if is_constant(node):
            coeffs[ONE] += coeff * constant_value(node)

        elif simple_fluent(node):
            params = node.args
            fluent = get_fluent_from_simple_fluent(node)
            fluent_repr = (fluent, params)
            coeffs[fluent_repr] = coeffs.get(fluent_repr, 0) + coeff

        elif is_operator(node):

            a, b = node.args
            if node.node_type in [OperatorKind.PLUS, OperatorKind.MINUS]:
                coeff_b = coeff
                if node.node_type == OperatorKind.MINUS:
                    coeff_b *= -1
                stack.append((a, coeff))
                stack.append((b, coeff_b))

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
        self.new_prob = None
        self.new_objects = {}

    def get_original_objects(self):
        self.objects = self.problem.all_objects
        return self.objects

    def createEmptyProblem(self) -> Problem:
        pass

    def createFluentMap(self):
        pass

    def createActionMap(self):
        pass

    def add_fluent_to_env(self, new_fluent):
        pass

class SAProblemSwapper(ProblemSwapper):

    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.new_prob = Problem()

    def createFluentMap(self):
        raise NotImplementedError

    def createActionMap(self):
        raise NotImplementedError

    def add_fluent_to_env(self, new_fluent, agent=None):
        raise NotImplementedError

class MAProblemSwapper(ProblemSwapper):
    problem: MultiAgentProblem  # class-level override

    def __init__(self, problem: MultiAgentProblem):
        super().__init__(problem)
        self.new_prob = MultiAgentProblem()
        self.action_map = OrderedDict()

    def add_fluent_to_env(self, new_fluent):
        self.new_prob.ma_environment.add_fluent(new_fluent)

    def createFluentMap(self):
        self.fluent_map['_env'] = OrderedDict()
        for fluent in self.problem.ma_environment.fluents:
            self.fluent_map['_env'][fluent.name] = {}
            self.fluent_map['_env'][fluent.name]['type'] = fluent.type
            self.fluent_map['_env'][fluent.name]['args'] = fluent.signature
        for agent in self.problem.agents:
            self.fluent_map[agent.name] = OrderedDict()
            for fluent in agent.fluents:
                self.fluent_map[agent.name][fluent.name] = {}
                self.fluent_map[agent.name][fluent.name]['type'] = fluent.type
                self.fluent_map[agent.name][fluent.name]['args'] = fluent.signature
        return self.fluent_map

    def createActionMap(self):
        self.action_map['_env'] = OrderedDict()
        self.action_map['_env']['_GOAL'] = OrderedDict()
        self.action_map['_env']['_GOAL']['parameters'] = []
        self.action_map['_env']['_GOAL']['precs'] = []
        self.action_map['_env']['_GOAL']['effs'] = []
        self.action_map['_env']['_INIT'] = OrderedDict()
        self.action_map['_env']['_INIT']['parameters'] = []
        self.action_map['_env']['_INIT']['precs'] = []
        self.action_map['_env']['_INIT']['effs'] = []

        for agent in self.problem.agents:
            self.action_map[agent.name] = OrderedDict()
            for action in agent.actions:
                self.action_map[agent.name][action.name] = OrderedDict()
                self.action_map[agent.name][action.name]['parameters']=action.parameters
                self.action_map[agent.name][action.name]['precs'] = action.preconditions
                self.action_map[agent.name][action.name]['effs'] = action.effects
            self.action_map[agent.name]['_GOAL'] = OrderedDict()
            self.action_map[agent.name]['_GOAL']['parameters'] = []
            self.action_map[agent.name]['_GOAL']['precs'] = [g for g in agent.public_goals]
            self.action_map[agent.name]['_GOAL']['effs'] = []
            self.action_map[agent.name]['_INIT'] = OrderedDict()
            self.action_map[agent.name]['_INIT']['parameters'] = []
            self.action_map[agent.name]['_INIT']['precs'] = []
            self.action_map[agent.name]['_INIT']['effs'] = []
        init_fluents = [(get_fluent_from_simple_fluent(i), tuple(str(a) for a in i.args), self.problem.initial_value(i)) for i in self.problem.initial_values]
        for fluent, fluent_args, value in init_fluents:
            if '.' in fluent:
                agent_name, fluent_name = fluent.split('.')
            else:
                agent_name, fluent_name = '_env', fluent
            self.action_map[agent_name]['_INIT']['effs'].append((fluent_name, fluent_args, value))

    def createEmptyActionMap(self):
        empty_action_map = OrderedDict()
        empty_action_map['_env'] = OrderedDict()
        empty_action_map['_env']['_GOAL'] = OrderedDict()
        empty_action_map['_env']['_GOAL']['parameters'] = []
        empty_action_map['_env']['_GOAL']['precs'] = []
        empty_action_map['_env']['_GOAL']['effs'] = []
        empty_action_map['_env']['_INIT'] = OrderedDict()
        empty_action_map['_env']['_INIT']['parameters'] = []
        empty_action_map['_env']['_INIT']['precs'] = []
        empty_action_map['_env']['_INIT']['effs'] = []
        for agent in self.problem.agents:
            empty_action_map[agent.name] = OrderedDict()
            for action in agent.actions:
                empty_action_map[agent.name][action.name] = OrderedDict()
                empty_action_map[agent.name][action.name]['parameters'] = []
                empty_action_map[agent.name][action.name]['precs'] = []
                empty_action_map[agent.name][action.name]['effs'] = []
            empty_action_map[agent.name]['_GOAL'] = OrderedDict()
            empty_action_map[agent.name]['_GOAL']['parameters'] = []
            empty_action_map[agent.name]['_GOAL']['precs'] = []
            empty_action_map[agent.name]['_GOAL']['effs'] = []
            empty_action_map[agent.name]['_INIT'] = OrderedDict()
            empty_action_map[agent.name]['_INIT']['parameters'] = []
            empty_action_map[agent.name]['_INIT']['precs'] = []
            empty_action_map[agent.name]['_INIT']['effs'] = []

        return empty_action_map

    def find_fluent(self, agent, fluent):
        name = str(fluent).split('(')[0]
        if name in self.fluent_map[agent]:
            return self.fluent_map[agent][name]
        elif name in self.fluent_map['_env']:
            return self.fluent_map[agent][name]
        raise Exception(f"Fluent {name} not found")

    def is_private(self, fluent, agent):
        return fluent in self.fluent_map[agent]

    def fluent_get_range(self, fluent, agent='_env'):
        if fluent == ONE[0]:
            return Fraction(1), Fraction(1)
        fluent_type = self.fluent_map[agent][fluent]['type']
        return fluent_type.lower_bound, fluent_type.upper_bound

    def get_fluent_args(self, fluent, agent='_env'):
        return self.fluent_map[agent][fluent]['args']


    def generate_fluent_name(self, row):
        """
        Generate a fluent string and deduplicated name from a row.
        Adds agent_name prefix if the fluent is private.
        """
        agent_name = row['agent']
        row = row.drop('agent')
        parts = ["__"]
        for col, val in row.items():
            if val == 0:
                continue
            sign = "P" if val > 0 else "M"
            parts.append(f"{sign}{abs(val)}__{(agent_name if self.is_private(col, agent_name) else 'ENV')}__{col.lower()}__")

        return "".join(parts)

    def get_new_fluent_range(self, row ):
        agent_name = row['agent']
        row = row.drop('agent')
        total_lower = 0
        total_upper = 0
        for fluent_name, val in row.items():
            fluent_name = remove_counter_suffixes(fluent_name)
            agent = agent_name if self.is_private(fluent_name, agent_name) else '_env'
            lower, upper = self.fluent_get_range(fluent_name, agent)
            total_lower += min(lower*val, upper*val)
            total_upper += max(lower * val, upper * val)

        return total_lower, total_upper

    def get_new_fluents_args(self, row ):
        agent_name = row['agent']
        row = row.drop('agent')
        args = []
        for fluent_name, val in row.items():
            if val == 0:
                continue
            if fluent_name.split('#')[-1].isdigit():
                fluent_name = remove_counter_suffixes(fluent_name)
            agent = agent_name if self.is_private(fluent_name, agent_name) else '_env'
            fluent_arg_types = [a.type for a in self.get_fluent_args(fluent_name, agent)]
            args += fluent_arg_types
        return args

def remove_counter_suffixes(col_name):
    if col_name.split('#')[-1].isdigit():
        return col_name.rsplit('#', 1)[0]
    return col_name


def create_preconditions_table(readable_action_map):
    column_counter = defaultdict(int)
    columns = ['agent', 'action', 'precondition_index', 'operator', 'value', 'args']
    for agent, actions in readable_action_map.items():
        for action, data in actions.items():
            for lin_prec, _, _ in data['precs']['lin']:
                # Count occurrences of each fluent in this precondition
                local_counter = defaultdict(int)
                for fluent in lin_prec:
                    fluent_name, _ = fluent
                    count = local_counter[fluent_name]
                    col_name = fluent_column_name(fluent_name, count)
                    local_counter[fluent_name] += 1
                    if col_name not in column_counter:
                        column_counter[col_name] = None  # add new column
    columns += sorted(column_counter.keys())  # alphabetical order

    df_precs = pd.DataFrame(columns=columns)

    rows = []
    for agent, actions in readable_action_map.items():
        for action, data in actions.items():
            for idx, (lin_prec, operator, value) in enumerate(data['precs']['lin']):
                row = {'agent': agent, 'action': action, 'precondition_index': idx,
                       'operator': operator_to_symbol(operator), 'value': value}
                new_fluent_args = []
                # Track local occurrences
                local_counter = defaultdict(int)
                for fluent in lin_prec:
                    fluent_name, fluent_args = fluent
                    new_fluent_args += fluent_args
                    count = local_counter[fluent_name]
                    col_name = fluent_column_name(fluent_name, count)
                    row[col_name] = lin_prec[fluent]
                    local_counter[fluent_name] += 1
                # Fill remaining columns with 0
                row['args'] = new_fluent_args
                for col in df_precs.columns:
                    if col not in row:
                        row[col] = 0
                rows.append(row)

    return normalize_dataframe(pd.concat([df_precs, pd.DataFrame(rows)], ignore_index=True),
                                   ignore_as_divisor_columns=['precondition_index', 'value'],
                                   dont_change_columns=['precondition_index'], operator_column='operator')


def create_effects_table(readable_action_map):
    rows = []
    for agent, actions in readable_action_map.items():
        for action, data in actions.items():
            for idx, (target_fluent, target_args, change) in enumerate(data['effs']['lin']):
                row = {'agent': agent, 'action': action, 'effect_index': idx,
                       'target_fluent': target_fluent, 'target_args':target_args, 'change': change}
                rows.append(row)
    return  pd.DataFrame(rows)


def parse_comparison_precondition(prec):
    if not prec_is_comparison(prec):
            raise Exception("Not a comparison precondition!")
    lin_node1, lin_node2 = prec.args
    lin1 = parse_lin(lin_node1)
    lin2 = parse_lin(lin_node2)
    prec_simple_fluents = set(lin1.keys()) | set(lin2.keys())
    operator = prec.node_type
    prec_fluent = {}
    for p in prec_simple_fluents:
        if p != ONE:
            prec_fluent[p] = lin1.get(p, 0) - lin2.get(p, 0)
    value = lin2.get(ONE, 0) - lin1.get(ONE, 0)
    return prec_fluent, operator, value

def parse_simple_boolean_precondition(prec):
    assert(prec.node_type == OperatorKind.FLUENT_EXP and str(prec.type)=="bool")
    return get_fluent_from_simple_fluent(prec), prec.args

def parse_simple_boolean_assignment_effect(eff):
    target_var, expr_node = eff.fluent, eff.value
    target_fluent = get_fluent_from_simple_fluent(target_var), target_var.args
    target_name, target_args = target_fluent
    value = eff.value
    assert value.node_type == OperatorKind.BOOL_CONSTANT
    return target_name, target_args, value

def parse_simple_change_effect(eff):

    if isinstance(eff, tuple):
        fluent, args, value = eff
        return fluent, args, value

    target_var, expr_node = eff.fluent, eff.value
    target_fluent = get_fluent_from_simple_fluent(target_var), target_var.args
    target_name, target_args = target_fluent
    # Try to parse the RHS as a linear expression
    try:
        lin_expr = parse_lin(expr_node)
    except Exception:
        raise Exception(f'Effect {target_var}={expr_node} is not convertible to numeric STRIPS.')
    if set(lin_expr.keys()) == {ONE, target_fluent} and lin_expr.get(target_fluent) == 1:
        return target_name, target_args, lin_expr[ONE]
    raise Exception(f'Effect {target_var}={expr_node} is not convertible to numeric STRIPS.')


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

    def separate_linearfrom_boolean_effects_and_preconditions(self):
        readable_action_map = self.swapper.createEmptyActionMap()
        for agent in self.swapper.action_map:
            for action in self.swapper.action_map[agent]:

                readable_action_map[agent][action]['precs'] = {'lin': [], 'bool': []}
                readable_action_map[agent][action]['effs'] = {'lin': [], 'bool': []}

                for prec in self.swapper.action_map[agent][action]['precs']:
                    prec_parsers = [
                        (parse_comparison_precondition, 'lin'),
                        (parse_simple_boolean_precondition, 'bool'),
                    ]

                    for func, prec_type in prec_parsers:
                        try:
                            parsed_prec = func(prec)
                            readable_action_map[agent][action]['effs'][prec_type].append(parsed_prec)
                            break
                        except Exception:
                            continue
                    else:
                        raise ValueError(f"Impossible to compile effect {prec} to numeric STRIPS.")


                for eff in self.swapper.action_map[agent][action]['effs']:
                    eff_parsers = [
                        (parse_simple_change_effect, 'lin'),
                        (parse_simple_boolean_assignment_effect, 'bool'),
                    ]

                    for func, eff_type in eff_parsers:
                        try:
                            parsed_eff = func(eff)
                            readable_action_map[agent][action]['effs'][eff_type].append(parsed_eff)
                            break
                        except Exception:
                            continue
                    else:
                        raise ValueError(f"Impossible to compile effect {eff} to numeric STRIPS.")
        return readable_action_map

    def is_private(self, fluent, agent):
        return fluent in self.swapper.fluent_map[agent]

    def empty_copy_action(self, action, agent)-> InstantaneousAction:
        params = {p.name: p.type for p in self.swapper.action_map[agent][action]['parameters']}
        return InstantaneousAction(action, **params)

    def transform(self, problem: Problem) -> Problem:
        """Transforms the given SNP problem into an RT-compliant version."""
        self.problem = problem
        is_multi_agent = isinstance(problem, MultiAgentProblem)
        if is_multi_agent:
            self.swapper = MAProblemSwapper(problem)
            for agent in self.problem.agents:
                self.swapper.new_prob.add_agent(Agent(agent.name, self.swapper.new_prob))
        else:
            self.swapper = SAProblemSwapper(problem)

        for o in problem.all_objects:
            self.swapper.new_prob.add_object(o.name, o.type)
            new_object = self.swapper.new_prob.object(o.name)
            self.swapper.new_objects[o.name] = new_object

        self.swapper.createFluentMap()
        self.swapper.createActionMap()

        linear_actions_map = self.separate_linearfrom_boolean_effects_and_preconditions()
        df_precs = create_preconditions_table(linear_actions_map)
        df_effs = create_effects_table(linear_actions_map)

        fluent_dict = {}
        fluent_vector_dict = {}
        fluents_used = []
        actions_dict: Dict[Tuple[str, str], InstantaneousAction] = {}

        for idx, row in df_precs.iterrows():

            fluent_vector = row.drop(
            ['action', 'precondition_index', 'operator', 'value', 'args','coeff'])
            new_fluent_name = self.swapper.generate_fluent_name(fluent_vector)
            lower_bound, higher_bound = self.swapper.get_new_fluent_range(fluent_vector)
            args = {f'p_{i}':p for i, p in enumerate(self.swapper.get_new_fluents_args(fluent_vector))}

            if not new_fluent_name in fluent_dict:
                fluent_dict[new_fluent_name] = Fluent(new_fluent_name, RealType(lower_bound, higher_bound), **args)
                fluent_vector_dict[new_fluent_name] = fluent_vector
                self.swapper.add_fluent_to_env(fluent_dict[new_fluent_name])

            fluents_used.append(new_fluent_name)
            fluent = fluent_dict[new_fluent_name]

            action_name, agent_name, operator, value, args = (df_precs.loc[idx, 'action'], df_precs.loc[idx, 'agent'],
                                                        df_precs.loc[idx, 'operator'], df_precs.loc[idx, 'value'],
                                                              df_precs.loc[idx, 'args'])
            if action_name == '_GOAL':
                args_objects = []
                for a in args:
                    new_arg = self.swapper.new_objects[str(a)]
                    args_objects.append(new_arg)
                precon = get_operator_as_function(operator)(fluent(*args_objects), value)
                self.swapper.new_prob.agent(agent_name).add_public_goal(precon)
            elif action_name == '_INIT':
                continue
            else:
                if (action_name, agent_name) not in actions_dict:
                    actions_dict[(action_name, agent_name)] = self.empty_copy_action(action_name, agent_name)
                action = actions_dict[(action_name, agent_name)]

                args_objects = []
                for a in args:
                    if str(a) in self.swapper.new_objects:
                        new_arg = self.swapper.new_objects[str(a)]
                    else:
                        new_arg = action.parameter(str(a))
                    args_objects.append(new_arg)
                precon = get_operator_as_function(operator)(fluent(*args_objects), value)
                action.add_precondition(precon)

        df_precs['new_fluents'] = fluents_used

        # Setting up effects
        for idx, row in df_effs.iterrows():
            agent_name = row['agent']
            action_name = row['action']
            target_fluent = row['target_fluent']
            target_args = row['target_args']
            change = row['change']

            if action_name == '_GOAL':
                continue
            # For now assume that it is numeric. If it is a linear formula over fluents,
            # we need to create the same formula using copies of the old fluents in the new problem and add this value.
            if change == 0: # This happens when the change is an effect with linear function as the change TODO: Remove assumption
                continue

            if (action_name, agent_name) not in actions_dict:
                actions_dict[(action_name, agent_name)] = self.empty_copy_action(action_name, agent_name)
            action = actions_dict[(action_name, agent_name)]

            for new_f in fluent_vector_dict:

                # For every fluent in the new problem, for every combination of parameters in the new fluent
                # change the value of this fluent by coeff(new_fluent, old_problem_fluent) * change
                # If fluent appears numerous times in the new_fluent:
                # for every appearance, designate it as the target fluent
                # and treat all other appearances as a regular sub-fluent.

                vector = fluent_vector_dict[new_f]
                agent_name = vector['agent']
                pure_vector = dict(vector.drop('agent'))
                sub_fluents_params = []
                is_changed=False
                for sub_fluent in pure_vector:
                    if remove_counter_suffixes(sub_fluent) == target_fluent and pure_vector[sub_fluent]!=0:
                        is_changed = True
                if not is_changed:
                    continue
                # This is a collection of possible parameters for every sub-fluent
                for sub_fluent, val in pure_vector.items():
                    if val == 0:
                        sub_fluents_params.append([])
                        continue
                    sub_fluent = remove_counter_suffixes(sub_fluent)
                    sub_fluents_agent = '_env' if not self.swapper.is_private(sub_fluent, agent_name) else agent_name
                    sub_fluent_args = self.swapper.get_fluent_args(sub_fluent, sub_fluents_agent)
                    possible_parameters = [list(self.swapper.new_prob.objects(a.type)) for a in sub_fluent_args]
                    sub_fluents_params.append(possible_parameters)
                for i, (sub_fluent, coeff) in enumerate(dict(pure_vector).items()):
                    if remove_counter_suffixes(sub_fluent) != target_fluent:
                        continue
                    params = sub_fluents_params.copy()
                    params[i] = [[a] for a in target_args]

                    params = [arg_options for p in params for arg_options in p]
                    for param_combination in itertools.product(*params):
                        changing_fluent = fluent_dict[new_f](*param_combination)
                        new_value = Plus(changing_fluent, Times(coeff, change))
                        if action_name == '_INIT':
                            self.swapper.new_prob.set_initial_value(changing_fluent, new_value)
                        else:
                            action.add_effect(changing_fluent, new_value)

        for (action_name, agent_name), action in actions_dict.items():
            if action_name == '_INIT' or action_name == '_GOAL':
                continue
            if agent_name == '_env':
                self.swapper.new_prob.add_action(action)
            else:
                self.swapper.new_prob.agent(agent_name).add_action(action)


        # TODO:
        #  1. Add initial values assignment.
        #  2. Add non linear preconditions and effects to actions (can assume simplicity, I think)
        #  3. Clean everything up and possibly remove swapper

        print(self.swapper.new_prob)
        return self.swapper.new_prob

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


