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

        self.enemy_ships = list()
        self.all_ships = list(initial_state['pirate_ships'].keys())
        self.map = initial_state["map"]
        self.base = initial_state["base"]

        row_range, col_range = np.arange(len(self.map)), np.arange(len(self.map[0]))
        locations = list(itertools.product(row_range, col_range))
        self.distances_dict = dict()

        for loc_i in locations:
            self.distances_dict[loc_i] = dict()
            for loc_j in locations:
                self.distances_dict[loc_i][loc_j] = get_min_dist(self.map, loc_i, loc_j)

        self.scores_actions_dict = {"collect": 100, "deposit": 500, "plunder": 50, "wait": 0}

    def act(self, state):
        actions = get_actions(state, self.player_number, self.simulator)
        all_actions_scores = [
            (action, sum(self.calculate_score(state, atomic_action_per_ship) for atomic_action_per_ship in action)) for
            action in actions]
        best_action = max(all_actions_scores, key=lambda act: act[1])[0]
        return best_action

    def calculate_score(self, state, action):
        if action[0] == 'wait':
            return self.scores_actions_dict[action[0]]
        elif action[0] in ['collect', 'deposit']:
            return self.scores_actions_dict[action[0]] * state['treasures'][action[2]][
                'reward']  # action reward X treasure reward
        elif action[0] == 'plunder':
            sum_reward = 0
            ship2 = action[2]
            for _, treasure_item in state['treasures'].items():  # could be more than 2 treasures on board
                if treasure_item['location'] in [ship2, state['pirate_ships'][ship2]['location']]:
                    sum_reward += treasure_item['reward']
            return self.scores_actions_dict[action[0]] * sum_reward

        # now we are left with sail actions

        collected_treasure_scores = list()
        uncollected_treasures_scores = list()
        uncollected_treasures_sum_reward = 0
        collected_treasures_sum_reward = 0
        capacity = 0

        for treasure_name, treasure_info in state['treasures'].items():
            pirate_ship = action[1]
            if treasure_info['location'] not in self.all_ships:  # treasures to collect
                uncollected_treasures_sum_reward += treasure_info['reward']
                next_location = action[2]
                row, col = treasure_info['location']
                if self.map[row][col] == 'I':
                    all_adjacent_tiles = [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
                    legal_tiles = [(adj_row, adj_col) for adj_row, adj_col in all_adjacent_tiles if
                                            0 <= adj_row < len(self.map) and 0 <= adj_col < len(self.map[0]) and
                                            self.map[adj_row][adj_col] != 'I'] # legal_adjacent_tiles
                else:
                    legal_tiles = [treasure_info['location']]
                ship_treasure_distance = min([self.distances_dict[next_location][loc] for loc in
                                              legal_tiles])
                if ship_treasure_distance != -1:
                    score = self.get_treasure_score(ship_treasure_distance, treasure_info['reward'])
                    uncollected_treasures_scores.append(score)
            elif treasure_info['location'] == pirate_ship:  # treasure is on cargo of pirate_ship
                capacity += 1
                collected_treasures_sum_reward += treasure_info['reward']
                next_location = action[2]
                distance_to_base = self.distances_dict[next_location][self.base]
                if distance_to_base != -1:
                    score = self.get_treasure_score(distance_to_base, treasure_info['reward'])
                    collected_treasure_scores.append(score)
        if len(uncollected_treasures_scores) > 0:
            uncollected_treasures_scores = [score / uncollected_treasures_sum_reward for score in
                                            uncollected_treasures_scores]
        if len(collected_treasure_scores) > 0:
            collected_treasure_scores = [score / collected_treasures_sum_reward for score in collected_treasure_scores]

        # calc risk
        risk_score = 0
        for _, marine_ship_details in state['marine_ships'].items():
            marine_locations_list = [marine_ship_details['path'][marine_ship_details["index"]]]
            if len(marine_ship_details['path']) > 0:
                if marine_ship_details["index"] < len(marine_ship_details['path']) - 1:
                    marine_locations_list.append(marine_ship_details['path'][marine_ship_details["index"] + 1])
                if marine_ship_details["index"] > 0:
                    marine_locations_list.append(marine_ship_details['path'][marine_ship_details["index"] - 1])

            if state['pirate_ships'][action[1]]['location'] in marine_locations_list:
                risk_score -= 1 / len(marine_locations_list)
                if state['pirate_ships'][action[1]]['capacity'] < 2:
                    risk_score -= 1000

        # if capacity == 1:
        #     all_treasure_scores = collected_treasure_scores + treasure_to_collect_scores
        #     return max(all_treasure_scores)
        if len(collected_treasure_scores) > 0:
            return max(collected_treasure_scores)
        if len(uncollected_treasures_scores) > 0:
            return max(uncollected_treasures_scores)
        return risk_score

    def get_treasure_score(self, distance, reward):
        max_distance = len(self.map) + len(self.map[0])  # - 1
        return (max_distance - distance) * reward


def get_min_dist(map, location_1, location_2):
    x1, y1 = location_1
    x2, y2 = location_2

    if map[x1][y1] == 'I' or map[x2][y2] == 'I':
        return -1

    explored_tiles = [[False for y in range(len(map[0]))] for x in range(len(map))]
    explored_tiles[x1][y1] = True

    class Tile:
        def __init__(self, x, y, distance=0):
            self.x = x
            self.y = y
            self.distance = distance

    tiles_set = {Tile(x1, y1)}
    while tiles_set:
        tile = tiles_set.pop()
        if (tile.x, tile.y) == location_2:
            return tile.distance

        for diff_row, diff_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = tile.x + diff_row, tile.y + diff_col
            if 0 <= new_row < len(map) and 0 <= new_col < len(map[0]) and map[new_row][new_col] != 'I' and not \
                    explored_tiles[new_row][new_col]:
                explored_tiles[new_row][new_col] = True
                tiles_set.add(Tile(new_row, new_col, tile.distance + 1))

    return -1


def get_actions(state, player, simulator):
    legal_actions = []
    all_actions = {}
    simulator.state = state
    for pirate_ship_name, pirate_ship in state['pirate_ships'].items():
        if pirate_ship['player'] == player:
            x, y = pirate_ship['location']
            new_locs = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

            # sail actions
            sail_actions = []
            for new_x, new_y in new_locs:
                if 0 <= new_x < len(state['map']) and 0 <= new_y < len(state['map'][0]) and state['map'][new_x][
                    new_y] != 'I':
                    sail_actions.append(("sail", pirate_ship_name, (new_x, new_y)))

            # deposit actions
            deposit_actions = []
            if state['map'][x][y] == 'B' and pirate_ship['capacity'] < 2:
                for treasure_name, treasure in state['treasures'].items():
                    if treasure['location'] == pirate_ship_name:
                        deposit_actions.append(("deposit", pirate_ship_name, treasure_name))

            # collect actions
            collect_actions = []
            if pirate_ship['capacity'] != 0:
                for treasure_name, treasure in state['treasures'].items():
                    if treasure['location'] in new_locs:
                        collect_actions.append(("collect", pirate_ship_name, treasure_name))

            # plunder actions
            plunder_actions = []
            for adv_name, adv in state['pirate_ships'].items():
                if adv['player'] != player and (x, y) == adv['location'] and adv['capacity'] < 2:
                    plunder_actions.append(("plunder", pirate_ship_name, adv_name))

            all_actions[pirate_ship_name] = [('wait',
                                              pirate_ship_name)] + sail_actions + collect_actions + deposit_actions + plunder_actions

    all_actions_product = list(itertools.product(*all_actions.values()))

    for actions in all_actions_product:
        if _is_action_mutex(actions):
            legal_actions.append(actions)
    return legal_actions


#################################### UCT #######################################################

class UCTNode:

    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action
        self.visits = 0
        self.score = 0
        self.avg_score = 0
        self.children = []

    def get_uct(self, ):
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
            children = [child for child in selected_node.children if
                        is_action_legal(simulator, child.action, self.player_number)]
            selected_node = max(children, key=lambda child: child.get_uct())
            simulator.apply_action(selected_node.action, self.player_number)
            actions = self.get_actions(simulator.state, 3 - self.player_number, simulator)
            random_action = random.choice(actions)
            simulator.apply_action(random_action, 3 - self.player_number)
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

                # sail actions
                sail_actions = []
                for new_x, new_y in new_locs:
                    if 0 <= new_x < len(state['map']) and 0 <= new_y < len(state['map'][0]) and state['map'][new_x][
                        new_y] != 'I':
                        sail_actions.append(("sail", pirate_ship_name, (new_x, new_y)))

                # deposit actions
                deposit_actions = []
                if state['map'][x][y] == 'B' and pirate_ship['capacity'] < 2:
                    for treasure_name, treasure in state['treasures'].items():
                        if treasure['location'] == pirate_ship_name:
                            deposit_actions.append(("deposit", pirate_ship_name, treasure_name))

                # collect actions
                collect_actions = []
                if pirate_ship['capacity'] != 0:
                    for treasure_name, treasure in state['treasures'].items():
                        if treasure['location'] in new_locs:
                            collect_actions.append(("collect", pirate_ship_name, treasure_name))

                # plunder actions
                plunder_actions = []
                for adv_name, adv in state['pirate_ships'].items():
                    if adv['player'] != player and (x, y) == adv['location'] and adv['capacity'] < 2:
                        plunder_actions.append(("plunder", pirate_ship_name, adv_name))

                all_actions[pirate_ship_name] = [('wait',
                                                  pirate_ship_name)] + sail_actions + collect_actions + deposit_actions + plunder_actions

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
        if simulator.state["pirate_ships"][pirate_1_name]["location"] != simulator.state["pirate_ships"][pirate_2_name][
            "location"]:
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

    players_pirates = [pirate for pirate in simulator.state['pirate_ships'].keys() if
                       simulator.state['pirate_ships'][pirate]['player'] == player]

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
