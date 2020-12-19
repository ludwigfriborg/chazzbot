import keras
import numpy as np
import json
import os
import argparse
import time
import math
import chess
import re
from flask import Flask, request, Response

from train_network_generator import train_network, evaluate_model

# from train_network import train_network, evaluate_model
from keras.models import Sequential, load_model
import sunfish

from standard_test import standard_test
from play_game import play_game
from play_sunfish import play_sunfish
from search_algorithm import search

# Just disables the warning, doesn't enable AVX/FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
app = Flask(__name__)
current_model = ""
c_model = None
nodes_explored = 0
max_depth = 10000


def inference_function(input):
    global c_model

    # Make predictions with moves, place into tuple (score, move), sort them
    return c_model.predict(input)


def load_inference_model(model_path):
    global c_model

    c_model = load_model(model_path)


# -----------------------------------------
# web-api
# -----------------------------------------
@app.route("/getmove", methods=["POST"])
def getmove():
    global c_model

    request.get_json(force=True)
    move = str(request.json["move"])
    fen = str(request.json["fen"])

    board = chess.Board(fen + " w - - 0 0")
    board.turn = True  # use fen later
    board.castling_rights = True
    move = (move, "invalid")

    if chess.Move.from_uci(move[0]) in board.legal_moves or board.is_castling(
        chess.Move.from_uci(move[0])
    ):
        board.push(chess.Move.from_uci(move[0]))

        if board.is_checkmate():
            move = (move[0], "Check mate - you win")
        else:
            if not c_model:
                load_inference_model("model/" + current_model + ".h5")

            move = predict(board.fen(), False, max_time=2)
            print(move)
            board.push(chess.Move.from_uci(move[0]))
            if board.is_checkmate():
                move = (move[0], "Check mate - chazzbot wins")

    res = Response(
        json.dumps([{"fen": board.fen(), "move": move[0], "explination": move[1]}]),
        mimetype="application/json",
    )
    res.headers["Access-Control-Allow-Origin"] = "*"
    return res


# -----------------------------------------
# main
# -----------------------------------------


def predict(fen, turn=False, max_time=2):
    """
    Given keras model and fennotation, and turn: returns the best
    scoring move as well as a descriptive text.

    The max_time indicates the number of seconds the search will
    go on until search will be stopped with the best result returned.
    """
    global max_depth
    board = chess.Board(fen)
    board.turn = turn  # use fen later

    print("***** predicting *****")
    s = time.time()
    tmp, nodes_explored = search(
        0,
        board.copy(),
        True,
        inference_function,
        depth=max_depth,
        timer=time.time() + max_time,
        max_depth=max_depth,
    )
    print("* prediction score:", tmp[0])
    print("* Time it took (in s):", time.time() - s)
    print("* Nodes explored:", nodes_explored)
    print("***********************")

    move = tmp[1]
    return (
        str(move),
        "Predicted move: "
        + str(move)
        + ". With accumulated prediction being: "
        + str(tmp[0]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chezzbot, chess move predictor using a regular feed forward network!"
    )
    parser.add_argument("-t", "--train", help="Trains model", nargs=1)
    parser.add_argument(
        "-st", "--standard-test", help="Tests model on some standard moves", nargs=1
    )
    parser.add_argument("-pg", "--play-game", help="plays quick game", nargs=1)
    parser.add_argument("-s", "--server", help="start flask service", nargs=1)
    parser.add_argument(
        "-sun", "--sunfish", help="plays a game against the sunfish ai", nargs=1
    )

    args = parser.parse_args()

    if args.train:
        train_network(args.train[0])

    if args.standard_test:
        load_inference_model("model/" + args.standard_test[0] + ".h5")
        standard_test(c_model, predict)

    if args.play_game:
        load_inference_model("model/" + args.play_game[0] + ".h5")
        play_game(predict)

    if args.server:
        current_model = args.server[0]
        app.run(host="0.0.0.0", port=5557)

    # This will not work unless you have the modified version of sunfish
    if args.sunfish:
        load_inference_model("model/" + args.sunfish[0] + ".h5")
        play_sunfish(predict)
