import unified_planning.model.operators
from unified_planning.shortcuts import *
from snp_to_rt import SNP_RT_Transformer


def get_linear_problem():

    # Creating problem ‘variables’
    battery_charge = Fluent("battery_charge", RealType(0, 100))
    x_loc = Fluent("x", RealType(0, 10))
    y_loc = Fluent("y", RealType(0, 10))

    # Creating actions
    up = InstantaneousAction("up")
    down = InstantaneousAction("down")
    left = InstantaneousAction("left")
    right = InstantaneousAction("right")

    up.add_precondition(GE(battery_charge,  10))
    up.add_precondition(LE(y_loc, 9))
    up.add_precondition(LE(Plus(y_loc, 1), x_loc))
    up.add_effect(y_loc, Plus(y_loc, 1))
    up.add_effect(battery_charge, Minus(battery_charge, 10))

    down.add_precondition(GE(battery_charge,  10))
    down.add_precondition(GE(y_loc, 1))
    down.add_effect(y_loc, Minus(y_loc, 1))
    down.add_effect(battery_charge, Minus(battery_charge, 10))

    left.add_precondition(GE(battery_charge,  10))
    left.add_precondition(GE(x_loc, 1))
    left.add_effect(x_loc, Minus(x_loc, 1))
    left.add_precondition(LE(y_loc, x_loc-1))
    left.add_effect(battery_charge, Minus(battery_charge, 10))

    right.add_precondition(GE(battery_charge,  10))
    right.add_precondition(LE(x_loc, 9))
    right.add_effect(x_loc, Plus(x_loc, 1))
    right.add_effect(battery_charge, Minus(battery_charge, 10))

    # Populating the problem with initial state and goals
    problem = Problem("robot")
    problem.add_fluent(battery_charge)
    problem.add_action(up)
    problem.add_action(down)
    problem.add_action(left)
    problem.add_action(right)

    problem.set_initial_value(x_loc, 0)
    problem.set_initial_value(y_loc, 0)
    problem.set_initial_value(battery_charge, 100)

    problem.add_goal(Equals(x_loc, 10))
    problem.add_goal(Equals(y_loc, 10))

    return problem


problem = get_linear_problem()
SRT = SNP_RT_Transformer()
SRT.transform(problem)
