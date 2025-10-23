import unified_planning.model.operators
from unified_planning.model.multi_agent import MultiAgentProblem, Agent

from unified_planning.shortcuts import *
from snp_to_rt import SNP_RT_Transformer


def get_linear_problem():

    # Creating problem ‘variables’
    battery_type = UserType('Battery')
    b1 = Object('b1', battery_type)
    battery_charge = Fluent("battery_charge", RealType(0, 100), battery=battery_type)
    rock_type = UserType('rock')
    rock = Object('Shiny', rock_type)
    robot_x_loc = Fluent("rob_x", RealType(0, 10))
    robot_y_loc = Fluent("rob_y", RealType(0, 10))
    holds = Fluent("holds", BoolType(), obj=rock_type)
    x_loc = Fluent("x", RealType(0, 10), obj=rock_type)
    y_loc = Fluent("y", RealType(0, 10), obj=rock_type)

    # Creating actions
    up = InstantaneousAction("up", battery=battery_type)
    down = InstantaneousAction("down", battery=battery_type)
    left = InstantaneousAction("left", battery=battery_type)
    right = InstantaneousAction("right", battery=battery_type)
    pickup = InstantaneousAction("pickup", obj=rock_type)
    switch_places = InstantaneousAction("switch-places", obj1=rock_type, obj2=rock_type)


    obj = pickup.parameter('obj')
    pickup.add_precondition(Equals(x_loc(obj), robot_x_loc))
    pickup.add_precondition(Equals(y_loc(obj), robot_y_loc))
    pickup.add_effect(holds(obj), True)

    battery=up.parameter('battery')
    up.add_precondition(GE(battery_charge(b1),  10))
    up.add_precondition(LE(robot_y_loc, 9))
    up.add_precondition(LE(Plus(robot_y_loc, 1), robot_x_loc))
    up.add_effect(robot_y_loc, Plus(robot_y_loc, 1))
    up.add_effect(battery_charge(battery), Minus(battery_charge(battery), 10))

    obj1=switch_places.parameter('obj1')
    obj2=switch_places.parameter('obj2')
    switch_places.add_precondition(LE(Minus(x_loc(obj1), x_loc(obj2)), 2))
    switch_places.add_precondition(LE(Minus(y_loc(obj1), y_loc(obj2)), 2))
#    switch_places.add_effect(x_loc(obj1), x_loc(obj2))
#    switch_places.add_effect(y_loc(obj1), y_loc(obj2))
#    switch_places.add_effect(x_loc(obj2), x_loc(obj1))
#    switch_places.add_effect(y_loc(obj2), y_loc(obj1))

    battery = down.parameter('battery')
    down.add_precondition(GE(battery_charge(battery),  10))
    down.add_precondition(GE(robot_y_loc, 1))
    down.add_effect(robot_y_loc, Minus(robot_y_loc, 1))
    down.add_effect(battery_charge(battery), Minus(battery_charge(battery), 10))

    battery = left.parameter('battery')
    left.add_precondition(GE(battery_charge(battery),  10))
    left.add_precondition(GE(robot_x_loc, 1))
    left.add_effect(robot_x_loc, Minus(robot_x_loc, 1))
    left.add_precondition(LE(robot_y_loc, robot_x_loc-1))
    left.add_effect(battery_charge(battery), Minus(battery_charge(battery), 10))

    battery = right.parameter('battery')
    right.add_precondition(GE(battery_charge(battery),  10))
    right.add_precondition(LE(robot_x_loc, 9))
    right.add_effect(robot_x_loc, Plus(robot_x_loc, 1))
    right.add_effect(battery_charge(battery), Minus(battery_charge(battery), 10))

    # Populating the problem with initial state and goals
    problem = MultiAgentProblem("robot")

    robot = Agent("robot", problem)
    robot.add_fluent(battery_charge)
    robot.add_fluent(robot_y_loc)
    robot.add_fluent(robot_x_loc)
    robot.add_fluent(holds)

    robot.add_action(up)
    robot.add_action(down)
    robot.add_action(left)
    robot.add_action(right)
    robot.add_action(pickup)
    robot.add_action(switch_places)

    problem.add_object(rock)
    problem.add_object(b1)
    problem.ma_environment.add_fluent(x_loc)
    problem.ma_environment.add_fluent(y_loc)

    problem.add_agent(robot)
    robot.add_public_goal(holds(rock))

    problem.set_initial_value(x_loc(rock), 10)
    problem.set_initial_value(y_loc(rock), 10)
    problem.set_initial_value(Dot(robot, robot_x_loc), 0)
    problem.set_initial_value(Dot(robot, robot_y_loc), 0)
    problem.set_initial_value(Dot(robot, battery_charge(b1)), 100)
    problem.set_initial_value(Dot(robot, holds(rock)), False)

    return problem


problem = get_linear_problem()
SRT = SNP_RT_Transformer()
SRT.transform(problem)
