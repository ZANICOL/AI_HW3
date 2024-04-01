import itertools
import time
import numpy as np
from simulator import Simulator
import random

IDS = ["Your IDS here"]



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

    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.visits = 0
        self.score = 0
        self.avg_score = 0
        self.children = []

    def get_uct(self,):
        if self.visits == 0:
            return float('inf')
        else:
            return self.avg_score + (4 * np.log(self.parent.visits) / self.visits) ** 0.5



class UCTTree:
    """
    A class for a Tree. not mandatory to use but may help you.
    """
    def __init__(self):
        raise NotImplementedError


class UCTAgent:
    def __init__(self, initial_state, player_number):
        self.ids = IDS
        self.player_number = player_number
        self.my_ships = []
        self.simulator = Simulator(initial_state)
        for ship_name, ship in initial_state['pirate_ships'].items():
            if ship['player'] == player_number:
                self.my_ships.append(ship_name)


    def selection(self, node, simulator):
        selected_node = node
        while len(selected_node.children):
            children = [child for child in selected_node.children if is_action_legal(simulator, child.action, self.player_number)]
            selected_node = max(children, key=lambda child: child.get_uct())
            simulator.apply_action(selected_node.action, self.player_number)
            actions = self.get_actions(simulator.state, 3-self.player_number, simulator)
            random_action = random.choice(actions)
            simulator.apply_action(random_action, 3-self.player_number)
        return selected_node


    def expansion(self, node, simulator):
        actions = self.get_actions(simulator.state, self.player_number, simulator)
        for action in actions:
            node.children.append(UCTNode(node, action))


    def simulation(self, simulator, player):
        if simulator.turns_to_go == 0:
            players_score = simulator.get_score()
            return players_score['player 1'] if player else players_score['player 2']

        actions = self.get_actions(simulator.state, player, simulator)
        action = random.choice(actions)
        while not is_action_legal(simulator, action, player):
            action = random.choice(actions)
        simulator.act(action, player)
        return self.simulation(simulator, 3 - player)

    def backpropagation(self, node, simulation_result):
        while node:
            node.visits += 1
            node.score += simulation_result
            node.avg_score = node.score / node.visits
            node = node.parent

    def act(self, state):
        root = UCTNode()
        start_time = time.time()
        while time.time() - start_time < 4.7:
            simulator = Simulator(state)
            new_node = self.selection(root, simulator)
            if simulator.turns_to_go != 0:
                self.expansion(new_node, simulator)
            score = self.simulation(simulator, self.player_number)
            self.backpropagation(new_node, score)
        possible_actions = self.get_actions(state, self.player_number, simulator)
        children = [child for child in root.children if child.action in possible_actions]
        max_score_child = max(children, key=lambda child: child.avg_score)
        return max_score_child.action


    def get_actions(self, state, player, simulator):
        legal_actions = []
        all_actions = {}
        simulator.state = state
        for pirate_ship_name, pirate_ship in state['pirate_ships'].items():
            if pirate_ship['player'] == player:
                x, y = pirate_ship['location']
                new_locs = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

                #sail actions
                sail_actions = []
                for new_x, new_y in new_locs:
                    if 0 <= new_x < len(state['map']) and 0 <= new_y < len(state['map'][0]) and state['map'][new_x][new_y] != 'I':
                        sail_actions.append(("sail", pirate_ship_name, (new_x, new_y)))

                #deposit actions
                deposit_actions = []
                if state['map'][x][y] == 'B' and pirate_ship['capacity'] < 2:
                    for treasure_name, treasure in state['treasures'].items():
                        if treasure['location'] == pirate_ship_name:
                            deposit_actions.append(("deposit", pirate_ship_name, treasure_name))


                #collect actions
                collect_actions = []
                if pirate_ship['capacity'] != 0:
                    for treasure_name, treasure in state['treasures'].items():
                        if treasure['location'] in new_locs:
                            collect_actions.append(("collect", pirate_ship_name, treasure_name))

                #plunder actions
                plunder_actions = []
                for adv_name, adv in state['pirate_ships'].items():
                    if adv['player'] != player and (x, y) == adv['location'] and adv['capacity'] < 2:
                        plunder_actions.append(("plunder", pirate_ship_name, adv_name))

                all_actions[pirate_ship_name] = [('wait', pirate_ship_name)] + sail_actions + collect_actions + deposit_actions + plunder_actions

        all_actions_product = list(itertools.product(*all_actions.values()))

        for actions in all_actions_product:
            if _is_action_mutex(actions):
                legal_actions.append(actions)
        return legal_actions

def _is_action_mutex(global_action):
    if len(set([a[1] for a in global_action])) != len(global_action):
        return False
    collect_actions = [a for a in global_action if a[0] == 'collect']
    if len(collect_actions) > 1:
        treasures_to_collect = set([a[2] for a in collect_actions])
        if len(treasures_to_collect) != len(collect_actions):
            return False
    return True


def is_action_legal(simulator, action, player):
    def _is_move_action_legal(move_action, player):
        pirate_name = move_action[1]
        if pirate_name not in simulator.state['pirate_ships'].keys():
            return False
        if player != simulator.state['pirate_ships'][pirate_name]['player']:
            return False
        l1 = simulator.state['pirate_ships'][pirate_name]['location']
        l2 = move_action[2]
        if l2 not in simulator.neighbors(l1):
            return False
        return True

    def _is_collect_action_legal(collect_action, player):
        pirate_name = collect_action[1]
        treasure_name = collect_action[2]
        if treasure_name not in simulator.state['treasures']:
            return False
        if player != simulator.state['pirate_ships'][pirate_name]['player']:
            return False
        # check adjacent position
        l1 = simulator.state['treasures'][treasure_name]['location']
        if simulator.state['pirate_ships'][pirate_name]['location'] not in simulator.neighbors(l1):
            return False
        # check ship capacity
        if simulator.state['pirate_ships'][pirate_name]['capacity'] <= 0:
            return False
        return True

    def _is_deposit_action_legal(deposit_action, player):
        pirate_name = deposit_action[1]
        treasure_name = deposit_action[2]
        if treasure_name not in simulator.state['treasures']:
            return False
        # check same position
        if player != simulator.state['pirate_ships'][pirate_name]['player']:
            return False
        if simulator.state["pirate_ships"][pirate_name]["location"] != simulator.base_location:
            return False
        if simulator.state['treasures'][treasure_name]['location'] != pirate_name:
            return False
        return True

    def _is_plunder_action_legal(plunder_action, player):
        pirate_1_name = plunder_action[1]
        pirate_2_name = plunder_action[2]
        if player != simulator.state["pirate_ships"][pirate_1_name]["player"]:
            return False
        if simulator.state["pirate_ships"][pirate_1_name]["location"] != simulator.state["pirate_ships"][pirate_2_name]["location"]:
            return False
        return True

    def _is_action_mutex(global_action):
        assert type(
            global_action) == tuple, "global action must be a tuple"
        # one action per ship
        if len(set([a[1] for a in global_action])) != len(global_action):
            return True
        # collect the same treasure
        collect_actions = [a for a in global_action if a[0] == 'collect']
        if len(collect_actions) > 1:
            treasures_to_collect = set([a[2] for a in collect_actions])
            if len(treasures_to_collect) != len(collect_actions):
                return True

        return False

    players_pirates = [pirate for pirate in simulator.state['pirate_ships'].keys() if simulator.state['pirate_ships'][pirate]['player'] == player]

    if len(action) != len(players_pirates):
        return False
    for atomic_action in action:
        # trying to act with a pirate that is not yours
        if atomic_action[1] not in players_pirates:
            return False
        # illegal sail action
        if atomic_action[0] == 'sail':
            if not _is_move_action_legal(atomic_action, player):
                return False
        # illegal collect action
        elif atomic_action[0] == 'collect':
            if not _is_collect_action_legal(atomic_action, player):
                return False
        # illegal deposit action
        elif atomic_action[0] == 'deposit':
            if not _is_deposit_action_legal(atomic_action, player):
                return False
        # illegal plunder action
        elif atomic_action[0] == "plunder":
            if not _is_plunder_action_legal(atomic_action, player):
                return False
        elif atomic_action[0] != 'wait':
            return False
    # check mutex action
    if _is_action_mutex(action):
        return False
    return True