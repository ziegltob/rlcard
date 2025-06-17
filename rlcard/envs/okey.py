import itertools
import time

from networkx import dfs_predecessors
import networkx as nx
import numpy as np
from rlcard.envs import Env
from rlcard.games.okey.constants import (
    SIZE_OF_COMBINATIONS,
    SIZE_OF_DECK,
    SIZE_OF_VISIBLE_CARDS,
    get_point_table,
    COLOR_TO_NUMBER,
    NUMBER_TO_COLOR,
    CARDS_TO_NUMBER,
    NUMBER_TO_CARDS,
    DIFFERENT_COLOR_COMBINATIONS,
    SAME_NUMBER_COMBINATIONS,
    SAME_COLOR_COMBINATIONS,
)
from rlcard.games.okey.play_okey import OkeyGame, build_game_tree


class OkeyEnv(Env):
    def __init__(self, config, max_steps=np.inf, seeding_range=None) -> None:
        self.seeding_range = seeding_range
        self.steps = 0
        self.max_steps = max_steps
        self.discard_streak = 0
        self.invalid_selection_streak = 0
        self.high_cards_discarded = 0
        self.scores = []
        self.score_per_turn = [0]
        self._last_mask = None
        self.num_players = 1
        self.last_actions = []
        self.last_game_states = []

        self.name = "okey"
        self.game = OkeyGame()
        super().__init__(config=config)

        self.valid_combinations = list(
            itertools.combinations(
                range(len(self.game.dealt_cards)), SIZE_OF_COMBINATIONS
            )
        )

        self.discard_action_id = SIZE_OF_DECK
        self.same_color_action_id = self.discard_action_id + len(
            SAME_COLOR_COMBINATIONS
        )
        self.same_number_action_id = self.same_color_action_id + len(
            SAME_NUMBER_COMBINATIONS
        )
        self.diff_color_action_id = self.same_number_action_id + len(
            DIFFERENT_COLOR_COMBINATIONS
        )

        self.end_game_action_id = self.game.get_num_actions() - 1

        # with action feature use: SIZE_OF_DECK + SIZE_OF_DECK + 3
        self.action_shape = [
            [self.game.get_num_actions()]
            for _ in range(self.num_players)
            # [SIZE_OF_DECK + SIZE_OF_DECK + 3] for _ in range(self.num_players)
        ]
        # +1 for score, +1 for num_cards_left, +1 for last_action, +1 for num_remaining_actions, not used rn: +SIZE_OF_DECK for last SIZE_OF_DECK amount actions
        # the format here needs to match the shape from _extract_state()
        self.state_shape = [
            [
                SIZE_OF_DECK
                + SIZE_OF_DECK
                + 1
                # + SIZE_OF_DECK
                + self.game.get_num_actions()
                + self.game.get_num_actions()
            ]
            for _ in range(self.num_players)
        ]

        self.done = False
        self.game_tree = None
        # self.state = self.reset()

    def action_to_id(self, action):
        if isinstance(action, str):
            return CARDS_TO_NUMBER[action]
        else:
            all_selections = (
                SAME_COLOR_COMBINATIONS
                + SAME_NUMBER_COMBINATIONS
                + DIFFERENT_COLOR_COMBINATIONS
            )
            return all_selections.index(action) + self.discard_action_id

    def _get_legal_actions(self):
        ideal_actions = None

        if ideal_actions is not None:
            return [self.action_to_id(action) for action in ideal_actions]

        all_selections = (
            SAME_COLOR_COMBINATIONS
            + SAME_NUMBER_COMBINATIONS
            + DIFFERENT_COLOR_COMBINATIONS
        )
        available_selections = self.game.get_current_valid_selections()
        selection_ids = [
            all_selections.index(selection) + self.discard_action_id
            for selection in available_selections
        ]
        valid_discard_cards = [CARDS_TO_NUMBER[card] for card in self.game.dealt_cards]
        # valid_actions = []
        # if len(self.game.remaining_selections) == 0:
        #     valid_actions = valid_discard_cards + selection_ids + [self.end_game_action_id]
        # else:
        valid_actions = valid_discard_cards + selection_ids
        return valid_actions

    def get_remaining_actions(self):
        available_selections = self.game.remaining_selections
        all_selections = (
            SAME_COLOR_COMBINATIONS
            + SAME_NUMBER_COMBINATIONS
            + DIFFERENT_COLOR_COMBINATIONS
        )
        selection_ids = [
            all_selections.index(selection) + self.discard_action_id
            for selection in available_selections
        ]
        valid_discard_cards = [
            CARDS_TO_NUMBER[card] for card in self.game.deck + self.game.dealt_cards
        ]
        # last id is for ending the game, which is always a valid turn
        valid_actions = (
            valid_discard_cards + selection_ids
        )  # + [self.end_game_action_id]
        return valid_actions

    def encode_cards(self, cards, size):
        plane = np.zeros(size, dtype=int)
        for card in cards:
            card_id = CARDS_TO_NUMBER[card]
            plane[card_id] = 1
        return plane

    # def get_perfect_information(self):
    #     state = {}
    #     state["dealt_cards"] = self.encode_cards(self.game.dealt_cards, SIZE_OF_DECK)
    #     state["deck"] = self.encode_cards(self.game.deck, SIZE_OF_DECK)
    #     state["legal_actions"] = self._get_legal_actions()
    #     return state

    # def get_action_feature(self, action):
    #     cards = self._decode_action(action)
    #     if len(cards) == 1:
    #         # return [self.encode_cards(list(cards), SIZE_OF_DECK)]
    #         # +3 for same_color, same_number, is_sequence
    #         return np.concatenate((self.encode_cards(list(cards), SIZE_OF_DECK), np.zeros(SIZE_OF_DECK + 3)))
    #     elif len(cards) == 3:
    #         # return [self.encode_cards(list(cards), SIZE_OF_DECK)]
    #         # return np.concatenate((np.zeros(SIZE_OF_DECK), self.encode_cards(list(cards), SIZE_OF_DECK)))
    #         # Get card properties
    #         numbers = [int(card[0]) for card in cards]
    #         colors = [card[1] for card in cards]

    #         # Encode relationships
    #         same_color = int(len(set(colors)) == 1)
    #         same_number = int(len(set(numbers)) == 1)
    #         is_sequence = int(sorted(numbers) == list(range(min(numbers), max(numbers) + 1)))

    #         # Combine features
    #         return np.concatenate((
    #             np.zeros(SIZE_OF_DECK),
    #             self.encode_cards(list(cards), SIZE_OF_DECK),
    #             [same_color, same_number, is_sequence]
    #         ))

    def encode_actions(self, actions):
        plane = np.zeros(self.game.get_num_actions(), dtype=int)
        for action in actions:
            plane[action] = 1
        return plane

    def has_sequence(self, cards, length=3):
        numbers = set([int(card[0]) for card in cards])
        counter = 0
        for number in sorted(list(numbers)):
            if number + 1 in numbers:
                counter += 1
                if counter >= length:
                    return 1
            else:
                counter = 0
        return 0

    def has_same_color_sequence(self, cards, length=3):
        for card in cards:
            number = int(card[0])
            color = card[1]
            if (
                number <= 6
                and f"{number+1}{color}" in cards
                and f"{number+2}{color}" in cards
            ):
                return 1
        return 0

    def _extract_state(self, state=None):
        dealt_cards = self.encode_cards(self.game.dealt_cards, SIZE_OF_DECK)
        remaining_actions = self.encode_actions(self.get_remaining_actions())
        score = [self.game.overall_score]
        # high_value_dealt_cards = [
        #     len([card for card in self.game.dealt_cards if int(card[0]) > 5])
        # ]
        # high_value_deck_cards = [
        #     len([card for card in self.game.deck if int(card[0]) > 5])
        # ]
        # has_sequence = [self.has_sequence(self.game.dealt_cards)]
        # has_same_color_sequence = [self.has_same_color_sequence(self.game.dealt_cards)]
        deck = self.encode_cards(self.game.deck, SIZE_OF_DECK)
        # score_diff = [self.score_per_turn[-1]]
        # num_remaining_selections = [len(self.game.remaining_selections)]
        # num_cards_left = [len(self.game.deck) + len(self.game.dealt_cards)]
        # take last 24 actions and pad them with -1
        # last_actions = self.last_actions + [-1] * (
        #     SIZE_OF_DECK - len(self.last_actions)
        # )
        last_actions = self.encode_actions(self.last_actions)
        obs = np.concatenate(
            (
                dealt_cards,  # index 0-23
                score,  # index 24
                last_actions,  # index 25-218
                remaining_actions,  # index 219-412
                deck,  # index 413-437
                # score_diff,
                # num_cards_left,
                # num_remaining_selections,
                # high_value_dealt_cards,
                # high_value_deck_cards,
                # has_sequence,
                # has_same_color_sequence,
            )
        )
        legal_actions = self._get_legal_actions()
        # featured_legal_actions = {}
        # for action in legal_actions:
        #     cards = self._decode_action(action)
        #     if len(cards) == 1:
        #         # featured_legal_actions[action] = [self.encode_cards(cards, SIZE_OF_DECK)]
        #         featured_legal_actions[action] = [self.encode_cards(cards, SIZE_OF_DECK), np.zeros(SIZE_OF_DECK)]
        #     elif len(cards) == 3:
        #         # featured_legal_actions[action] = [self.encode_cards(cards, SIZE_OF_DECK)]
        #         featured_legal_actions[action] = [np.zeros(SIZE_OF_DECK), self.encode_cards(cards, SIZE_OF_DECK)]
        return {
            "obs": obs,
            # "legal_actions": featured_legal_actions,
            # "legal_actions": {action: self.encode_cards(list(self._decode_action(action)), SIZE_OF_DECK) for action in legal_actions},
            "legal_actions": {action: None for action in legal_actions},
            "raw_obs": obs,
            "raw_legal_actions": legal_actions,
        }

    def _decode_action(self, action_id):
        if action_id in range(self.discard_action_id):
            card = NUMBER_TO_CARDS[action_id]
            return [card]
        elif action_id in range(self.discard_action_id, self.same_color_action_id):
            combo_id = action_id - self.discard_action_id
            card_combination = SAME_COLOR_COMBINATIONS[combo_id]
            return card_combination
        elif action_id in range(self.same_color_action_id, self.same_number_action_id):
            combo_id = action_id - self.same_color_action_id
            card_combination = SAME_NUMBER_COMBINATIONS[combo_id]
            return card_combination
        elif action_id in range(self.same_number_action_id, self.diff_color_action_id):
            combo_id = action_id - self.same_number_action_id
            card_combination = DIFFERENT_COLOR_COMBINATIONS[combo_id]
            return card_combination
        # elif action_id == self.end_game_action_id:
        # return "END_GAME"
        else:
            raise Exception(f"decode_action: unknown action_id={action_id}")

    def is_card_valid(self, card_number, card_color):
        return 0 < card_number <= 8 and 1 <= card_color <= 3

    # turn action into a card
    def numbers_to_card(self, card_number, color_number):
        if self.is_card_valid(card_number, color_number):
            return str(card_number) + NUMBER_TO_COLOR[color_number]
        else:
            return "invalid"

    # turn a card string like "1r" to the (number, color) tuple represented with numbers only
    def card_to_numbers(self, card):
        if (
            len(card) != 2
            or card[1] not in COLOR_TO_NUMBER
            or not 1 <= int(card[0]) <= 8
        ):
            return -1, -1  # invalid card
        number = int(card[0])
        color_number = COLOR_TO_NUMBER[card[1]]
        return number, color_number

    def get_ideal_action(self, tree_depth=2):
        # TODO probably this should return more actions
        # so the model learns what to select and not just limit all
        # actions when deck < 5
        if self.game.is_done():
            return None
        tree_depth = 1 if tree_depth <= 0 else tree_depth
        if self.game_tree is None:
            start = time.time()
            self.game_tree = build_game_tree(self.game, tree_depth)
            end = time.time()
            # print(f"Duration: {end - start} seconds")
        elif self.game in self.game_tree:
            # descendants = nx.descendants(self.game_tree, self.game)
            # descendants = descendants.union({self.game})
            # self.game_tree = self.game_tree.subgraph(descendants)
            # TODO it seems like recreating the tree solves the issue that some
            # parts of the tree are not played until is_done
            # this could have a huge performance boost if subtree is used
            self.game_tree = build_game_tree(self.game, tree_depth)
        else:
            print(
                "the game is not in the tree, but the tree is not none"
                + "this is due to the shuffle issue when building the tree"
            )
        # find leaf with highest score
        leaf_nodes = [
            node
            for node in self.game_tree.nodes
            if self.game_tree.out_degree(node) == 0
        ]
        if len(leaf_nodes) == 0:
            return None
        if not self.game in self.game_tree:
            print("debug")
        # max_score_leaf = max(leaf_nodes, key=lambda node: node.overall_score)
        best_leaves = sorted(
            leaf_nodes, key=lambda game: game.overall_score, reverse=True
        )[:5]
        best_moves = []
        for leaf in best_leaves:
            nodes_to_root = nx.shortest_path(
                self.game_tree, source=self.game, target=leaf
            )
            # 0 is root and 1 is the next step taken towards max_score_leaf
            edge_data = self.game_tree.get_edge_data(self.game, nodes_to_root[1])
            best_moves.append(edge_data["move"])
        return best_moves

    def action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=np.int32)
        valid_actions = self._get_legal_actions()

        for action_id in valid_actions:
            mask[action_id] = 1

        self._last_mask = mask
        return np.array(mask)

    # def step(self, action, player_id):
    def step(self, action, raw_action=False):
        old_score = self.game.overall_score
        action_cards = self._decode_action(action)
        # if self.last_actions[0] == -1:
        #     self.last_actions = [action]
        # else:
        self.last_actions.append(action)
        # if action_cards == "END_GAME":
        #     self.game.end_game()
        # when single card, then discard action
        if len(action_cards) == 1:
            discard_card = action_cards[0]
            if discard_card in self.game.dealt_cards:
                self.store_last_game()
                self.game.discard_card(discard_card)
            else:
                print(
                    f"Invalid discard card {discard_card}, dealt cards: {self.game.dealt_cards}"
                )
        # if multiple cards, selection action
        elif len(action_cards) == 3:
            selected_cards = action_cards
            # check if move is valid
            if (
                all(card in self.game.dealt_cards for card in selected_cards)
                and not any(card == "invalid" for card in selected_cards)
                and len(set(selected_cards)) == 3
            ):
                self.store_last_game()
                is_valid = self.game.select_cards(set(selected_cards))
                if not is_valid:
                    print(
                        f"Invalid selection {selected_cards}, dealt cards: {self.game.dealt_cards}"
                    )
            else:
                print(
                    f"Should never be here. Invalid selection {selected_cards}, dealt cards: {self.game.dealt_cards}"
                )
        else:
            print(f"Unknown action: {action}")

        score_diff = self.game.overall_score - old_score
        self.score_per_turn.append(score_diff)

        observation = self._extract_state()
        done = self.game.is_done()
        if done:
            self.scores.append(self.game.overall_score)
            last_scores = self.scores[-1000:]
            if len(self.scores) % 1000 == 0:
                # print(f"Average score {sum(last_scores)/len(last_scores)}")
                # print(
                #     f"Times >=300: {len([score for score in last_scores if score >= 300 < 400])/len(last_scores)}"
                # )
                # print(
                #     f"Times >=400: {len([score for score in last_scores if score >= 400])/len(last_scores)}"
                # )
                # print overall metrics:
                print(f"Average score {sum(self.scores)/len(self.scores)}")
                print(
                    f"Times >=300: {len([score for score in self.scores if score >= 300 < 400]) / len(self.scores)}"
                )
                print(
                    f"Times >=400: {len([score for score in self.scores if score >= 400]) / len(self.scores)}"
                )
        self.steps += 1
        return observation, self.get_player_id()
        # return observation

    def is_over(self):
        return len(self.game.dealt_cards) == 0

    def get_player_id(self):
        return 0

    # this is only called when the game is over
    def get_payoffs(self):
        # score_diff = self.score_per_turn[-1]
        if self.game.is_over():
            if self.game.overall_score >= 400:
                return np.array([1])
            # elif self.game.overall_score >= 390:
            #     return np.array([0.8])
            # elif self.game.overall_score >= 380:
            #     return np.array([0.7])
            # elif self.game.overall_score >= 370:
            #     return np.array([0.6])
            # elif self.game.overall_score >= 350:
            #     return np.array([0.5])
            elif self.game.overall_score >= 300:
                return np.array([0.3])
            else:
                # return np.array([self.game.overall_score / 1500])
                return np.array([-1])
                if len(self.game.remaining_selections) > 0:
                    return np.array([-3])
                else:
                    return np.array([-1])
        # return np.array([score_diff / 10000])
        return np.array([-1])

    def reset(self, seed=None, options=None):
        self.game = OkeyGame(seed=seed, seeding_range=self.seeding_range)
        self.steps = 0
        self.discard_streak = 0
        self.invalid_selection_streak = 0
        self.high_cards_discarded = 0
        self.score_per_turn = [0]
        player_id = 0
        observation = self._extract_state()
        self.game_tree = None
        self.last_actions = []
        self.last_game_states = []

        return observation, player_id

    def store_last_game(self):
        pass
        # uncomment code below if storing game states is required
        # game_copy = OkeyGame(
        #     deck=list(self.game.deck),
        #     dealt_cards=list(self.game.dealt_cards),
        #     disposed_cards=set(self.game.disposed_cards),
        #     score=self.game.overall_score,
        #     remaining_selections=self.game.remaining_selections,
        # )
        # self.last_game_states.append(game_copy)

    def step_back(self):
        # print("stepping back")
        if len(self.last_game_states) > 0:
            self.game = self.last_game_states.pop()
            self.steps -= 1
            return True
        return False
