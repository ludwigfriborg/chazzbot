import chess
import sunfish
import re


def play_sunfish(prediction_function):
    """
    predict (function) - should have input params:
        fen notation (string), turn (bool), max searchtime seconds (int)
    """
    current_board = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    sunfish_board = sunfish.Position(
        sunfish.initial, 0, (True, True), (True, True), 0, 0
    )
    sunfish_searcher = sunfish.Searcher()
    search_time = 2
    count = 0

    while not current_board.is_game_over():
        print(current_board)

        if not current_board.is_valid():
            sunfish.print_pos(sunfish_board)
            print("Invalid board..")
            break

        prediction = prediction_function(
            current_board.fen(), True, max_time=search_time
        )  # play as white

        if prediction[0] == "":
            break

        if not chess.Move.from_uci(prediction[0]) in current_board.pseudo_legal_moves:
            print("Invalid move..")
            break

        c = chess.Move.from_uci(prediction[0])
        right_place = (
            chess.square_rank(c.to_square) == 7 or chess.square_rank(c.to_square) == 0
        )
        right_piece = (
            str(current_board.piece_at(c.from_square)) == "p"
            or str(current_board.piece_at(c.from_square)) == "P"
        )
        if right_piece and right_place:
            print("Promoted")
            c.promotion = 5
        current_board.push(c)

        count += 1

        print(count, "Chazzbot move: ", prediction[0])
        if current_board.is_checkmate():
            print("Chazzbot won")
            break
        print(current_board)

        pred_list = list(prediction[0])
        match = re.match("([a-h][1-8])" * 2, prediction[0])
        move = sunfish.parse(match.group(1)), sunfish.parse(match.group(2))

        # sunfish make move
        sunfish_board = sunfish_board.move(move)
        sunfish_move, sunfish_score = sunfish_searcher.search(
            sunfish_board, secs=search_time
        )
        sunfish_board = sunfish_board.move(sunfish_move)
        sunfish_move_adjusted = sunfish.render(119 - sunfish_move[0]) + sunfish.render(
            119 - sunfish_move[1]
        )

        count += 1
        print(count, "Sunfish move: ", sunfish_move_adjusted)
        print("Sunfish score: ", sunfish_score)

        c = chess.Move.from_uci(sunfish_move_adjusted)
        right_place = (
            chess.square_rank(c.to_square) == 7 or chess.square_rank(c.to_square) == 0
        )
        right_piece = (
            str(current_board.piece_at(c.from_square)) == "p"
            or str(current_board.piece_at(c.from_square)) == "P"
        )
        if right_piece and right_place:
            print("Promoted")
            c.promotion = 5
        current_board.push(c)

        if current_board.is_checkmate():
            print("Sunfish won")
            print(current_board)
            break

        if not current_board.is_valid():
            print("Sunfish broke it...")

        if current_board.is_checkmate():
            break

    print("Final score (chazzbot-sunfish): ", current_board.result())
