import itertools

SIZE_OF_COMBINATIONS = 3
SIZE_OF_VISIBLE_CARDS = 5
SIZE_OF_DECK = 24

CARDS = {
    "1r",
    "2r",
    "3r",
    "4r",
    "5r",
    "6r",
    "7r",
    "8r",
    "1b",
    "2b",
    "3b",
    "4b",
    "5b",
    "6b",
    "7b",
    "8b",
    "1y",
    "2y",
    "3y",
    "4y",
    "5y",
    "6y",
    "7y",
    "8y",
}

CARDS_TO_NUMBER = {
    "1r": 0,
    "2r": 1,
    "3r": 2,
    "4r": 3,
    "5r": 4,
    "6r": 5,
    "7r": 6,
    "8r": 7,
    "1b": 8,
    "2b": 9,
    "3b": 10,
    "4b": 11,
    "5b": 12,
    "6b": 13,
    "7b": 14,
    "8b": 15,
    "1y": 16,
    "2y": 17,
    "3y": 18,
    "4y": 19,
    "5y": 20,
    "6y": 21,
    "7y": 22,
    "8y": 23,
}
NUMBER_TO_CARDS = {
    0: "1r",
    1: "2r",
    2: "3r",
    3: "4r",
    4: "5r",
    5: "6r",
    6: "7r",
    7: "8r",
    8: "1b",
    9: "2b",
    10: "3b",
    11: "4b",
    12: "5b",
    13: "6b",
    14: "7b",
    15: "8b",
    16: "1y",
    17: "2y",
    18: "3y",
    19: "4y",
    20: "5y",
    21: "6y",
    22: "7y",
    23: "8y",
}

COLOR_TO_NUMBER = {"b": 1, "r": 2, "y": 3}
NUMBER_TO_COLOR = {1: "b", 2: "r", 3: "y"}

POINT_TABLE = None
SORTED_POINT_TABLE_KEYS = None

SAME_NUMBER_COMBINATIONS = [
    ["1b", "1r", "1y"],
    ["2b", "2r", "2y"],
    ["3b", "3r", "3y"],
    ["4b", "4r", "4y"],
    ["5b", "5r", "5y"],
    ["6b", "6r", "6y"],
    ["7b", "7r", "7y"],
    ["8b", "8r", "8y"],
]
SAME_COLOR_COMBINATIONS = [
    ["1r", "2r", "3r"],
    ["2r", "3r", "4r"],
    ["3r", "4r", "5r"],
    ["4r", "5r", "6r"],
    ["5r", "6r", "7r"],
    ["6r", "7r", "8r"],
    ["1b", "2b", "3b"],
    ["2b", "3b", "4b"],
    ["3b", "4b", "5b"],
    ["4b", "5b", "6b"],
    ["5b", "6b", "7b"],
    ["6b", "7b", "8b"],
    ["1y", "2y", "3y"],
    ["2y", "3y", "4y"],
    ["3y", "4y", "5y"],
    ["4y", "5y", "6y"],
    ["5y", "6y", "7y"],
    ["6y", "7y", "8y"],
]
NUMBER_COMBINATIONS: list[tuple[str, str, str]] = [
    ("1", "2", "3"),
    ("2", "3", "4"),
    ("3", "4", "5"),
    ("4", "5", "6"),
    ("5", "6", "7"),
    ("6", "7", "8"),
]
COLORS = ["r", "b", "y"]
DIFFERENT_COLOR_COMBINATIONS = []


def get_point_table():
    global POINT_TABLE
    if POINT_TABLE is None:
        init_point_table()
    return POINT_TABLE


def get_sorted_point_table_keys():
    global SORTED_POINT_TABLE_KEYS
    if SORTED_POINT_TABLE_KEYS is None:
        init_point_table()
    return SORTED_POINT_TABLE_KEYS


def init_point_table():
    global POINT_TABLE, SORTED_POINT_TABLE_KEYS, DIFFERENT_COLOR_COMBINATIONS
    different_combinations = []
    POINT_TABLE = {}
    SAME_NUMBER_BASE_SCORE = 20
    SAME_COLOR_BASE_SCORE = 50
    RANDOM_COLOR_BASE_SCORE = 10
    SCORE_INCREMENT = 10

    # add all permutations for same number combinations
    for index, number_sequence in enumerate(SAME_NUMBER_COMBINATIONS):
        score = SAME_NUMBER_BASE_SCORE + (index * SCORE_INCREMENT)
        for permutation in itertools.permutations(number_sequence):
            POINT_TABLE[",".join(permutation)] = score
    # print(len(POINT_TABLE))

    # add all permutations for same color number combinations
    for index, number_sequence in enumerate(NUMBER_COMBINATIONS):
        score = SAME_COLOR_BASE_SCORE + (index * SCORE_INCREMENT)
        for color in COLORS:
            combination_with_color = [number + color for number in number_sequence]
            for permutation in itertools.permutations(combination_with_color):
                POINT_TABLE[",".join(permutation)] = score
    # print(len(POINT_TABLE))

    # add all permutations for number sequence with random color
    for index, number_sequence in enumerate(NUMBER_COMBINATIONS):
        score = RANDOM_COLOR_BASE_SCORE + (index * SCORE_INCREMENT)
        all_color_combinations = list(
            itertools.product(COLORS, repeat=len(number_sequence))
        )
        # filter out combinations that all have the same color, like [1r, 2r, 3r]
        # first filter out all_color_combinations that don't have at least 2 unique values
        # [r, r, r] filtered out, [r, b, r] is included
        # then zip the remaining color combinations with the sequence [1, 2, 3]
        filtered_combinations = [
            [f"{n}{s}" for n, s in zip(number_sequence, combo)]
            for combo in all_color_combinations
            if len(set(combo)) > 1
        ]
        different_combinations.append(filtered_combinations)
        for combination in filtered_combinations:
            for permutation in itertools.permutations(combination):
                POINT_TABLE[",".join(list(permutation))] = score

    DIFFERENT_COLOR_COMBINATIONS += [
        combo for sublist in different_combinations for combo in sublist
    ]
    SORTED_POINT_TABLE_KEYS = sorted(POINT_TABLE.keys())
    print(f"POINT_TABLE initialized: {len(POINT_TABLE)}")


def main():
    init_point_table()


if __name__ == "__main__":
    print("Run constants main()")
    main()
