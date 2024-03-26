IDS = ["Your IDS here"]
from simulator import Simulator
import random
from numpy import log

#test
class Agent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.player_number = player_number
        self.my_ships = []
        self.simulator = Simulator(initial_state)
        for ship_name, ship in initial_state['pirate_ships'].items():
            if ship['player'] == player_number:
                self.my_ships.append(ship_name)

    def act(self, state):
        raise NotImplementedError



class UCTNode:
    """
    A class for a single node. not mandatory to use but may help you.
    """

    def __init__(self, parent=None, action=None,state=None):
        self.parent = parent
        self.children = list()
        self.wins = 0
        self.trys = 0
        self.action = action
        self.state = state
        # raise NotImplementedError



class UCTTree:
    """
    A class for a Tree. not mandatory to use but may help you.
    """

    def __init__(self):
        self.root = UCTNode()
        # raise NotImplementedError


class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.player_number = player_number
        self.my_ships = []
        self.simulator = Simulator(initial_state)
        for ship_name, ship in initial_state['pirate_ships'].items():
            if ship['player'] == player_number:
                self.my_ships.append(ship_name)

    def selection(self, UCT_tree): # check again
        node_set = {UCT_tree.root}
        max_node = None
        cur_max_score = 0
        while node_set != None:
            node_i = node_set.pop()
            if len(node_i.children) >= len(self.act(node_i)):
                continue
            node_set.add(node_i.children)
            temp_score = (node_i.wins / node_i.trys) + (2 * log(node_i.parent.trys) / node_i.trys) ** 0.5
            # if node_i.trys != 0:
            #     temp_score =
            # TO-DO maybey + -
            if cur_max_score < temp_score:
                cur_max_score = temp_score
                max_node = node_i

        return max_node


    def expansion(self, UCT_tree, parent_node):
        whole_actions = self.act(parent_node.state)
        new_state =
        for action_i in whole_actions
        current_node = UCTNode(parent_node,parent_node)


        raise NotImplementedError

    def simulation(self):
        raise NotImplementedError

    def backpropagation(self, simulation_result):
        raise NotImplementedError

    def act(self, state):
        actions = {}
        self.simulator.set_state(state)
        collected_treasures = []
        for ship in self.my_ships:
            actions[ship] = set()
            neighboring_tiles = self.simulator.neighbors(state["pirate_ships"][ship]["location"])
            for tile in neighboring_tiles:
                actions[ship].add(("sail", ship, tile))
            if state["pirate_ships"][ship]["capacity"] > 0:
                for treasure in state["treasures"].keys():
                    if state["pirate_ships"][ship]["location"] in self.simulator.neighbors(
                            state["treasures"][treasure]["location"]) and treasure not in collected_treasures:
                        actions[ship].add(("collect", ship, treasure))
                        collected_treasures.append(treasure)
            for treasure in state["treasures"].keys():
                if (state["pirate_ships"][ship]["location"] == state["base"]
                        and state["treasures"][treasure]["location"] == ship):
                    actions[ship].add(("deposit", ship, treasure))
            for enemy_ship_name in state["pirate_ships"].keys():
                if (state["pirate_ships"][ship]["location"] == state["pirate_ships"][enemy_ship_name]["location"] and
                        self.player_number != state["pirate_ships"][enemy_ship_name]["player"]):
                    actions[ship].add(("plunder", ship, enemy_ship_name))
            actions[ship].add(("wait", ship))

        while True:
            whole_action = []
            for atomic_actions in actions.values():
                for action in atomic_actions:
                    if action[0] == "deposit":
                        whole_action.append(action)
                        break
                    if action[0] == "collect":
                        whole_action.append(action)
                        break
                else:
                    whole_action.append(random.choice(list(atomic_actions)))
            whole_action = tuple(whole_action)
            if self.simulator.check_if_action_legal(whole_action, self.player_number):
                return whole_action
