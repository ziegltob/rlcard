import copy
import random
import time
import itertools
from collections import Counter, deque
from rlcard.games.okey.constants import (
    CARDS,
    DIFFERENT_COLOR_COMBINATIONS,
    SAME_COLOR_COMBINATIONS,
    SAME_NUMBER_COMBINATIONS,
    get_point_table,
    CARDS_TO_NUMBER,
    SIZE_OF_VISIBLE_CARDS,
    SIZE_OF_COMBINATIONS,
    SIZE_OF_DECK,
    COLOR_TO_NUMBER,
    get_sorted_point_table_keys,
)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class OkeyGame:
    def __init__(
        self,
        deck=None,
        dealt_cards=None,
        disposed_cards=None,
        score=0,
        seed=None,
        seeding_range=None,
        remaining_selections=None,
    ) -> None:
        self.deck = deck if deck is not None else []
        self.disposed_cards = disposed_cards if disposed_cards is not None else set()
        self.dealt_cards = dealt_cards if dealt_cards is not None else []
        self.overall_score = score
        self.point_table = None
        self.point_table = get_point_table()
        self.seed = seed
        self.seeding_range = seeding_range
        self.game_over = False

        if len(self.deck) == 0 and len(self.dealt_cards) == 0:
            self.get_shuffled_deck(CARDS)
            self.deal_cards()

        self.valid_combinations = list(
            itertools.combinations(range(len(self.dealt_cards)), SIZE_OF_COMBINATIONS)
        )
        if remaining_selections is None:
            self.remaining_selections = self.get_all_selections()
        else:
            self.remaining_selections = remaining_selections

    def get_num_players(self):
        return 1
    
    def end_game(self):
        self.game_over = True

    def get_num_actions(self):
        return (
            SIZE_OF_DECK
            + len(SAME_COLOR_COMBINATIONS)
            + len(SAME_NUMBER_COMBINATIONS)
            + len(DIFFERENT_COLOR_COMBINATIONS)
            # + 1 # action for ending the game
        )

    def get_shuffled_deck(self, deck_of_cards: set[str]) -> list[str]:
        self.deck = list(deck_of_cards)
        if self.seed is not None:
            self.deck = sorted(self.deck)
            random.seed(self.seed)
        elif self.seeding_range is not None:
            seed = random.randint(0, self.seeding_range)
            random.seed(seed)
        random.shuffle(self.deck)
        # print("New shuffled deck created")

    def deal_cards(self):
        if self.deck is None:
            print("Deck not initialized")
            return
        # print("Dealing new cards")
        while len(self.dealt_cards) < SIZE_OF_VISIBLE_CARDS:
            if len(self.deck) == 0:
                # print("No more cards in deck")
                break
            card = self.deck.pop()
            # print(f"Dealing card {card}")
            self.dealt_cards.append(card)
        # print(f"Dealt Cards: {self.dealt_cards}")

    def get_current_valid_selections(self):
        valid_card_selections = []
        possible_combinations = list(
            itertools.combinations(range(len(self.dealt_cards)), SIZE_OF_COMBINATIONS)
        )
        for combination in possible_combinations:
            cards_for_combination = [self.dealt_cards[i] for i in combination]
            if ",".join(cards_for_combination) in self.point_table:
                valid_card_selections.append(sorted(cards_for_combination))
        return valid_card_selections

    def get_all_selections(self):
        valid_card_selections = []
        all_cards = self.deck + self.dealt_cards
        possible_combinations = list(
            itertools.combinations(range(len(all_cards)), SIZE_OF_COMBINATIONS)
        )
        for combination in possible_combinations:
            cards_for_combination = [all_cards[i] for i in combination]
            if ",".join(cards_for_combination) in self.point_table:
                valid_card_selections.append(sorted(cards_for_combination))
        return valid_card_selections

    def select_cards(self, selected_cards: set[str]):
        # validation of selected cards
        if len(selected_cards) != 3:
            print(f"Please select 3 cards: {selected_cards}")
            return
        if not selected_cards.issubset(set(self.dealt_cards)):
            print("Only select cards from the dealt cards")
            return
        if self.game_over:
            print("Game Over! Restart the Game")
            return
        score = self.validate_combination(selected_cards)
        if score == -1:
            # print("Combination does not exist")
            return False
        # execute the valid selection
        self.execute_selection(selected_cards, score)
        return True

    def execute_selection(self, selected_cards: set[str], score: int):
        if (
            not selected_cards.issubset(set(self.dealt_cards))
            or len(selected_cards) != 3
        ):
            print("Can't execute faulty card selection")
            return
        self.overall_score += score
        # print(f"Scored {score} points, overall score {self.overall_score} points")
        self.dealt_cards = [
            card for card in self.dealt_cards if card not in selected_cards
        ]
        self.disposed_cards.update(selected_cards)
        self.remaining_selections = [
            selection
            for selection in self.remaining_selections
            if not any(card in selection for card in selected_cards)
        ]
        self.deal_cards()

    def validate_combination(self, selected_cards: set[str]) -> int:
        combination = ",".join(list(selected_cards))
        if combination in self.point_table:
            return self.point_table[combination]
        else:
            return -1

    def discard_card(self, card):
        if self.game_over:
            print("Game Over! Restart the Game.")
            return
        if card in self.dealt_cards:
            # print(f"Discarding {card}")
            self.dealt_cards.remove(card)
            self.disposed_cards.add(card)
            self.remaining_selections = [
                selection
                for selection in self.remaining_selections
                if not card in selection
            ]
            self.deal_cards()
        else:
            print(f"Card not in dealt cards! Card {card} can not be discarded")
            return

    def cards_to_number(self, cards):
        # return [CARDS_TO_NUMBER[card] for card in cards]
        return [int(card[0]) for card in cards]

    def cards_to_color(self, cards):
        return [COLOR_TO_NUMBER[card[1]] for card in cards]

    # return the number of points for all selectable combinations (3 out of 5 = 10 possible combinations)
    # returns -1 if a combination is not valid
    def get_points_for_combinations(self) -> list[int]:
        points_list = []
        for combination in self.valid_combinations:
            combination_cards = [
                self.dealt_cards[card_index]
                for card_index in combination
                if card_index < len(self.dealt_cards)
            ]
            combination_key = ",".join(combination_cards)
            score = (
                self.point_table[combination_key]
                if combination_key in self.point_table
                else -1
            )
            points_list.append(score)
        return points_list

    def card_nr(self, card):
        return int(card[0])

    def is_sequence_in_cards(self, cards, sequence_length=3):
        sorted_cards = sorted(cards)
        # numbers = sorted(int(card[1]) for card in cards)
        is_sequence_in_dealt_cards = 0
        is_sequence_same_color = 0
        sequence = []
        count = 1
        for i in range(1, len(sorted_cards)):
            if self.card_nr(sorted_cards[i]) == self.card_nr(sorted_cards[i - 1]):
                continue
            elif self.card_nr(sorted_cards[i]) == self.card_nr(sorted_cards[i - 1]) + 1:
                count += 1
                sequence.append(sorted_cards[i])
                if count >= sequence_length:
                    is_sequence_in_dealt_cards = 1
                    is_sequence_same_color = (
                        1 if len(set([card[1] for card in sequence])) == 1 else 0
                    )
            else:
                sequence = []
                count = 1

        return is_sequence_in_dealt_cards, is_sequence_same_color

    def is_triple_number_available(self):
        numbers = [int(card[0]) for card in self.dealt_cards]
        counter = Counter(numbers)
        return 1 if any(count >= 3 for count in counter.values()) else 0

    def get_observation(self):
        # check if there is a sequence in dealt_cards
        is_sequence_in_dealt_cards, is_sequence_same_color = self.is_sequence_in_cards(
            self.dealt_cards
        )

        dealt_cards_padding_size = max(SIZE_OF_VISIBLE_CARDS - len(self.dealt_cards), 0)
        deck_pad_size = max(SIZE_OF_DECK - len(self.deck), 0)

        observation = np.concatenate(
            [
                # np.pad(
                #     np.array(self.cards_to_number(self.dealt_cards), dtype=np.int32),
                #     (0, dealt_cards_padding_size),
                #     constant_values=-1,
                # ),
                # np.pad(
                #     np.array(self.cards_to_color(self.dealt_cards), dtype=np.int32),
                #     (0, dealt_cards_padding_size),
                #     constant_values=-1,
                # ),
                np.pad(
                    np.array(
                        [CARDS_TO_NUMBER[card] for card in self.dealt_cards],
                        dtype=np.int32,
                    ),
                    (0, dealt_cards_padding_size),
                    constant_values=-1,
                ),
                # np.array(self.get_points_for_combinations()),
                np.pad(
                    np.array(
                        sorted([CARDS_TO_NUMBER[card] for card in self.deck]),
                        dtype=np.int32,
                    ),
                    (0, deck_pad_size),
                    constant_values=-1,
                ),
                np.array([self.overall_score]),
                np.array([is_sequence_in_dealt_cards]),
                np.array([is_sequence_same_color]),
                np.array([self.is_triple_number_available()]),
                # TODO add
                #  available_points or something to give future vision
                #  current_steps
                #  add dead cards that can no longer gain points
                #  indicate potential of cards, e.g. when cards already have a sequence of 2
                # TODO add a matrix of each rem
            ]
        )
        return observation

    def get_state(self, player_id):
        state = {}
        if not self.is_done():
            state["dealt_cards"] = self.dealt_cards
            state["deck"] = self.deck
            state["score"] = self.overall_score
        return state

    def is_done(self):
        return len(self.dealt_cards) == 0
        return self.game_over

    def is_over(self):
        return self.is_done()

    def do_action(self, action):
        if isinstance(action, str):
            self.discard_card(action)
        elif isinstance(action, list) or isinstance(action, tuple):
            self.select_cards(action)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, OkeyGame)
            and set(self.dealt_cards) == set(other.dealt_cards)
            and set(self.deck) == set(other.deck)
            and self.overall_score == other.overall_score
        )

    def __hash__(self) -> int:
        return hash(
            (frozenset(self.dealt_cards), frozenset(self.deck), self.overall_score)
        )


def build_game_tree(game: OkeyGame, depth, tree=None):
    if game.is_done() or depth == 0:
        return
    if tree == None:
        tree = nx.DiGraph()
    selection_moves = game.get_current_valid_selections()
    discard_moves = game.dealt_cards
    all_moves = selection_moves + discard_moves
    # for move in selection_moves:
    for move in all_moves:
        if len(game.deck) > 0:
            new_game = None
            if isinstance(move, str):
                for i, card in enumerate(game.deck):
                    new_game = OkeyGame(
                        deck=list(game.deck),
                        dealt_cards=list(game.dealt_cards),
                        disposed_cards=set(game.disposed_cards),
                        score=game.overall_score,
                        remaining_selections=game.remaining_selections,
                    )
                    # shift deck around for each iteration to create a new game for each possible card drawn
                    d = deque(new_game.deck)
                    d.rotate(-i)
                    new_game.deck = list(d)
                    new_game.discard_card(move)
                    tree.add_edge(game, new_game, move=move)
                    if new_game in tree:
                        continue
                    if not new_game.is_done():
                        build_game_tree(new_game, depth - 1, tree)
            else:
                # generate all 3 card combinations that are possible with the deck
                combinations = list(itertools.combinations(game.deck, 3))
                if len(combinations) == 0:
                    new_game = OkeyGame(
                        deck=list(game.deck),
                        dealt_cards=list(game.dealt_cards),
                        disposed_cards=set(game.disposed_cards),
                        score=game.overall_score,
                        remaining_selections=game.remaining_selections,
                    )
                    new_game.select_cards(set(move))
                    tree.add_edge(game, new_game, move=move)
                    if new_game in tree:
                        continue
                    if not new_game.is_done():
                        build_game_tree(new_game, depth - 1, tree)
                else:
                    for combination in combinations:
                        new_game = OkeyGame(
                            deck=list(game.deck),
                            dealt_cards=list(game.dealt_cards),
                            disposed_cards=set(game.disposed_cards),
                            score=game.overall_score,
                            remaining_selections=game.remaining_selections,
                        )
                        sorted_deck = [
                            card for card in new_game.deck if card not in combination
                        ] + list(combination)
                        new_game.deck = sorted_deck
                        new_game.select_cards(set(move))
                        tree.add_edge(game, new_game, move=move)
                        if new_game in tree:
                            continue
                        if not new_game.is_done():
                            build_game_tree(new_game, depth - 1, tree)
        else:
            new_game = OkeyGame(
                deck=list(game.deck),
                dealt_cards=list(game.dealt_cards),
                disposed_cards=set(game.disposed_cards),
                score=game.overall_score,
                remaining_selections=game.remaining_selections,
            )
            if isinstance(move, str):
                new_game.discard_card(move)
            else:
                new_game.select_cards(set(move))
            tree.add_edge(game, new_game, move=move)
            if new_game in tree:
                continue
            if not new_game.is_done():
                build_game_tree(new_game, depth - 1, tree)

    return tree

def has_sequence(cards, length=3):
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

def get_same_color_sequence(cards, length=3):
    found_combos = []
    for card in sorted(cards):
        number = int(card[0])
        color = card[1]
        number_range = 8 - (length - 1)
        if (number <= number_range and all(f"{i}{color}" in cards for i in range(number, number + length))):
            found_combos.append([f"{i}{color}" for i in range(number, number + length)])
    return found_combos

def get_potential_same_color_sequences(cards):
    potential_sequence_length = 2
    sequence_length = 3
    found_combos = []
    for card in sorted(cards):
        number = int(card[0])
        color = card[1]
        cards_in_potential_sequence = [f"{i}{color}" for i in range(number, number + sequence_length) if f"{i}{color}" in cards]
        if len(cards_in_potential_sequence) >= potential_sequence_length:
            found_combos.append(cards_in_potential_sequence)
    return found_combos

def get_same_number_combination(cards):
    numbers = [int(card[0]) for card in cards]
    counter = Counter(numbers)
    return [num for num, count in counter.items() if count >= 3]

def play_678_strategy():
    scores = list()

    for i in range(3000):
        done = False
        game = OkeyGame()
        disposable_cards = []
        while not done:
            done = game.is_done()
            if done:
                break
            
            disposable_card = "1b"
            for card in game.dealt_cards:
                if not any(card in sel for sel in game.remaining_selections):
                    disposable_cards.append(card)
            
            same_color_sequences = get_same_color_sequence(game.dealt_cards)
            has_same_color_sequence = len(same_color_sequences) > 0
            starting_number = int(same_color_sequences[0][0][0]) if has_same_color_sequence else 0
            color = same_color_sequences[0][0][1] if has_same_color_sequence else ""
            if has_same_color_sequence and (starting_number == 6 or starting_number <= 3):
                selection = set([f"{starting_number}{color}", f"{starting_number+1}{color}", f"{starting_number+2}{color}"])
                # print(f"selection: {selection}")
                # print(f"dealt cards: {game.dealt_cards}")
                game.select_cards(selection)
                continue
            
            if has_same_color_sequence and starting_number >= 5:
                selection = set([f"{starting_number}{color}", f"{starting_number+1}{color}", f"{starting_number+2}{color}"])
                if all(card[0] in ["6", "7", "8"] for card in game.dealt_cards if card not in selection):
                    game.select_cards(selection)
                    disposable_card = f"{starting_number+3}{color}"
                    if not any(disposable_card in sel for sel in game.remaining_selections):
                        disposable_cards.append(disposable_card)
                    continue
            
            same_number_combos = get_same_number_combination(game.dealt_cards)
            if same_number_combos and same_number_combos[0] in [5, 8]:
                selection = set([f"{same_number_combos[0]}{color}" for color in ("r", "b", "y")])
                game.select_cards(selection)
                continue
            
            if any(card in game.dealt_cards for card in disposable_cards):
                card_to_discard = next(card for card in disposable_cards if card in game.dealt_cards)
                game.discard_card(card_to_discard)
                continue
            
            # in case no 678 same color combination is available, discard any non 678
            if all(card[0] in ["6", "7", "8"] for card in game.dealt_cards):
                sequences_678 = get_potential_same_color_sequences(game.dealt_cards)
                if sequences_678:
                    discardable = [card for card in game.dealt_cards if not any(card in seq for seq in sequences_678)]
                    if discardable:
                        game.discard_card(random.choice(discardable))
                    else:
                        game.discard_card(random.choice(game.dealt_cards))
                else:
                    game.discard_card(random.choice(game.dealt_cards))
            else:
                small_cards = [card for card in game.dealt_cards if card[0] not in ["6", "7", "8"]]
                small_sequences = get_potential_same_color_sequences(small_cards)
                if len(small_sequences) > 0:
                    discardable_cards = [card for card in small_cards if not any(card in seq for seq in small_sequences)]
                    if discardable_cards:
                        game.discard_card(random.choice(discardable_cards))
                    else:
                        game.discard_card(random.choice(small_cards))
                else:
                    for card in game.dealt_cards:
                        if not any(number in card for number in ["6", "7", "8"]):
                            # TODO check if there is a 3 and prevent discarding a 3 if there are also other small cards that could be discarded
                            game.discard_card(card)
                            break
        done = False
        scores.append(game.overall_score)
    print(f"Average score {sum(scores)/len(scores)}")
    print(f"Times >=300: {len([score for score in scores if 300 <= score < 400]) / len(scores)}")
    print(f"Times >=400: {len([score for score in scores if 400 <= score]) / len(scores)}")


def play_okey_game(seed=None):
    okey_game = OkeyGame(seed=seed)
    # start = time.time()
    # tree = build_game_tree(okey_game, 3)
    # print("tree ready", tree.number_of_nodes())
    # end = time.time()
    # print(f"Duration: {end - start} seconds")
    return
    # pos = nx.spring_layout(tree)
    # nx.draw(tree, pos, node_color='lightblue', node_size=2000)
    # plt.title("Graph with Custom Objects")
    # plt.show()
    return

    print("Game initialized")
    print("[S]elect 3 cards with input: s1y,1r,1b")
    print("[D]iscard a card with input: d1y")
    # TODO implement game end:
    #  there are less than 3 cards available
    #  there is no more combination possible
    while len(okey_game.dealt_cards) > 0:
        print("Dealt Cards: ", okey_game.dealt_cards)
        user_input = input(">>> ")
        if len(user_input) == 0 or user_input[0] not in ["s", "d"]:
            print(
                "Please select or discard cards, example inputs: 's1y,1r,1b' or 'd1y'"
            )
            continue
        if user_input[0] == "s":
            parsed_input = user_input[1:].split(",")
            selected_cards = set(parsed_input)
            okey_game.select_cards(selected_cards)
            print(f"Score: {okey_game.overall_score}")
        elif user_input[0] == "d":
            parsed_input = user_input[1:]
            card = parsed_input
            okey_game.discard_card(card)
        else:
            print("Could not parse user input")


if __name__ == "__main__":
    # play_okey_game()
    play_678_strategy()
