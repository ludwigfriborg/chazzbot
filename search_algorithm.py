import math
import time
from data_extractor import convert_fen_label, reshape_moves
import numpy as np


def search(
    score,
    board,
    maximizing,
    inference_function,
    depth=1,
    a_i=-math.inf,
    b_i=math.inf,
    timer=math.inf,
    add_random=0,
    max_depth=10000,
):
    """
    Searches for the best move further down in the search tree
    The depth defines how far the search tree will be searched
    Returns an prediction score.

    Using Minimax algorithim with alpha beta pruning

    note: increasing depth seriously improves performance of
    estimations but increases the prediction time drastically

    Recommended: add_random if used should perhaps be 0.05 or in that range
    """
    nodes_explored = 0

    # If its time to stop searching or max depth reached return node, can only be done when comming from a maximizing node
    current_time = time.time()
    if (depth < 1 or current_time > timer) and not maximizing:
        nodes_explored += 1
        return score, nodes_explored

    a, b = a_i, b_i
    tmp = (-math.inf, "") if maximizing else (math.inf, "")

    fen_before = board.fen()
    prediction_inputs = []
    prediction_moves = []

    # Generate all possible moves as inputs for predictions
    for legal in board.legal_moves:
        board.push(legal)
        if board.is_checkmate():
            return (math.inf if maximizing else -math.inf, legal)
        else:
            input_thing = reshape_moves(
                convert_fen_label(fen_before, False),
                convert_fen_label(board.fen(), True),
            )
            prediction_inputs.append(input_thing)
            prediction_moves.append(legal)
        board.pop()

    # If no possible moves we have lost or won, abort
    if len(prediction_inputs) == 0:
        return (-math.inf, "") if maximizing else (math.inf, ""), nodes_explored

    # Make predictions with moves, place into tuple (score, move), sort them
    ps = inference_function(np.array(prediction_inputs))

    # generate random gaussian curve with max at add_random and add it to ps
    if add_random != 0:
        g = np.random.normal(0, add_random, len(ps))
        ps = ps + g

    predictions = zip(ps, prediction_moves)
    predictions = sorted(predictions, key=lambda x: x[0], reverse=True)

    # only explore three best
    for p in predictions[0:4]:
        if maximizing:
            board.push(p[1])
            predicted, nexp = search(
                p[0],
                board.copy(),
                False,
                inference_function,
                depth=depth - 1,
                a_i=a,
                b_i=b,
                timer=timer,
                max_depth=max_depth,
            )
            board.pop()

            nodes_explored += nexp

            if depth == max_depth:
                print("* max", predicted[0])

            if tmp[0] <= predicted[0]:
                tmp = (predicted[0], p[1])

            a = max(a, tmp[0])
            if b <= a:
                return tmp, nodes_explored

        else:
            board.push(p[1])
            predicted, nexp = search(
                p[0],
                board.copy(),
                True,
                inference_function,
                depth=depth - 1,
                a_i=a,
                b_i=b,
                timer=timer,
                max_depth=max_depth,
            )
            board.pop()

            nodes_explored += nexp

            if tmp[0] >= predicted[0]:
                tmp = (predicted[0], p[1])

            b = min(b, tmp[0])
            if b <= a:
                return tmp, nodes_explored

    return tmp, nodes_explored