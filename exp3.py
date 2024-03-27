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
            children = [child for child in selected_node.children]
            selected_node = max(children, key=lambda child: child.get_uct())
            simulator.apply_action(selected_node.action, self.player_number)
            actions = self.get_actions(simulator.get_state(), 3-self.player_number, simulator)
            random_action = random.choice(actions)
            simulator.apply_action(random_action, 3-self.player_number)
        return selected_node


    def expansion(self, node, simulator):
        actions = self.get_actions(simulator.get_state(), self.player_number, simulator)
        for action in actions:
            node.children.append(UCTNode(node, action))


    def simulation(self, simulator, player):
        if simulator.turns_to_go == 0:
            players_score = simulator.get_score()
            return players_score['player 1'] if player else players_score['player 2']

        actions = self.get_actions(simulator.get_state(), player, simulator)
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
        while time.time() - start_time < 4.8:
            simulator = Simulator(state)
            new_node = self.selection(root, simulator)
            if simulator.turns_to_go != 0:
                self.expansion(new_node, simulator)
            score = self.simulation(simulator, self.player_number)
            self.backpropagation(new_node, score)
        #for child in root.children:
        #    print(child.action, child.score, child.avg_score)
        max_score_child = max(root.children, key=lambda child: child.avg_score)
        print(state)
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


