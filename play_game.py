import chess


def play_game(prediction_function):
    current_board = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    count = 0

    while not current_board.is_game_over():
        print(current_board)
        prediction = prediction_function(
            current_board.fen(), current_board.turn, max_time=2
        )
        count += 1

        print(str(count) + ". " + prediction[1])

        if not current_board.is_valid():
            print("Invalid board..")
            break
        if not chess.Move.from_uci(prediction[0]) in current_board.legal_moves:
            print("Invalid move..")
            break

        current_board.push(chess.Move.from_uci(prediction[0]))
    print("Final score: ", current_board.result())
