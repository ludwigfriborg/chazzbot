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

from data_extractor import convert_fen_label, reshape_moves
from train_network_generator import train_network, evaluate_model
# from train_network import train_network, evaluate_model
from keras.models import Sequential, load_model
import sunfish

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)
current_model = ''
c_model = False
nodes_explored = 0
max_depth = 100

#-----------------------------------------
# web-api
#-----------------------------------------
@app.route("/getmove", methods=['POST'])
def getmove():
  global c_model

  request.get_json(force=True)
  move = str(request.json['move'])
  fen = str(request.json['fen'])

  board = chess.Board(fen + ' w - - 0 0')
  board.turn = True # use fen later
  board.castling_rights = True
  move = (move, 'invalid')

  if chess.Move.from_uci(move[0]) in board.legal_moves or board.is_castling(chess.Move.from_uci(move[0])):
    board.push(chess.Move.from_uci(move[0]))

    if board.is_checkmate():
      move = (move[0], 'Check mate - you win')
    else:
      if not c_model:
        c_model = load_model('model/' + current_model + '.h5')

      move = predict(board.fen(), c_model, False, max_time=2)
      print(move)
      board.push(chess.Move.from_uci(move[0]))
      if board.is_checkmate():
        move = (move[0], 'Check mate - chazzbot wins')

  res = Response(json.dumps([{"fen": board.fen(), "move": move[0], "explination": move[1]}]),  mimetype='application/json')
  res.headers['Access-Control-Allow-Origin'] = '*'
  return res  


#-----------------------------------------
# main
#-----------------------------------------
def predict_depth(board, model, maximizing, depth=1, a_i=-math.inf, b_i=math.inf, timer=math.inf):
  '''
  Searches for the best move further down in the search tree
  The depth defines how far the search tree will be searched
  Returns an prediction score.

  Using Minimax algorithim with alpha beta pruning

  note: increasing depth seriously improves performance of 
  estimations but increases the prediction time drastically
  '''
  global nodes_explored, max_depth

  a, b = a_i, b_i
  tmp = (-math.inf, '') if maximizing else (math.inf, '')
  fen_before = board.fen()

  if board.is_checkmate():
    return(math.inf if maximizing else -math.inf, '')

  prediction_inputs = []
  prediction_moves = []
  for legal in board.legal_moves:
    board.push(legal)
    if board.is_checkmate():
      return (math.inf if maximizing else -math.inf, legal)
    else:
      input_thing = reshape_moves(convert_fen_label(fen_before, False), convert_fen_label(board.fen(), True))
      prediction_inputs.append(input_thing)
      prediction_moves.append(legal)
    board.pop()

  ps = model.predict(np.array(prediction_inputs))
  predictions = zip(ps, prediction_moves)
  predictions = sorted(predictions, key=lambda x: x[0], reverse=True)

  current_time = time.time()
  if (depth < 1 or current_time > timer) and maximizing:
    nodes_explored += 1
    return predictions[0]


  # only explore three best
  for p in predictions[0:4]:
    if maximizing:
      board.push(p[1])
      predicted = predict_depth(board.copy(), model, False, depth=depth-1, a_i=a, b_i=b, timer=timer)
      score = predicted[0] #+ p[0]
      board.pop()
      
      if depth == max_depth:
        print('* max', score)

      if tmp[0] < score:
        tmp = (score, p[1])

      a = max(a, tmp[0])
      if b <= a:
        return tmp

    else:
      board.push(p[1])
      predicted = predict_depth(board.copy(), model, True, depth=depth-1, a_i=a, b_i=b, timer=timer)
      score = predicted[0] #- p[0]
      board.pop()

      if tmp[0] > score:
        tmp = (score, p[1])

      b = min(b, tmp[0])
      if b <= a:
         return tmp

  return tmp

import time
def predict(fen, model, turn=False, max_time=2):
  '''
  Given keras model and fennotation, and turn: returns the best
  scoring move as well as a descriptive text.

  The max_time indicates the number of seconds the search will 
  go on until search will be stopped with the best result returned.
  '''
  global nodes_explored, max_depth
  board = chess.Board(fen)
  board.turn = turn # use fen later

  print('***** predicting *****')
  nodes_explored = 0
  s = time.time()
  tmp = predict_depth(board.copy(), model, True, depth=max_depth, timer=int(time.time()) + max_time)
  print('* Time it took (in s):', time.time() - s)
  print('* Nodes explored:', nodes_explored)
  print('***********************')
  
  move = tmp[1]
  return str(move), 'Predicted move: ' + str(move) + '. With accumulated prediction being: ' + str(tmp[0])

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Chezzbot, chess move predictor using a regular feed forward network!')
  parser.add_argument('-t','--train', help='Trains model', nargs=1)
  parser.add_argument('-st','--standard-test', help='Tests model on some standard moves', nargs=1)
  parser.add_argument('-pg','--play-game', help='plays quick game', nargs=1)
  parser.add_argument('-s','--server', help='start flask service', nargs=1)
  parser.add_argument('-sun','--sunfish', help='plays a game against the sunfish ai', nargs=1)

  args = parser.parse_args()

  if args.train:
    train_network(args.train[0])

  if args.standard_test:
    model = load_model('model/' + args.standard_test[0] + '.h5')

    input_thing = reshape_moves(convert_fen_label('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 0', False), convert_fen_label('rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w - - 0 0', True))
    input_thing = np.array([input_thing])
    s = time.time()
    model.predict(input_thing)[0][0]
    print('Time it took (in s):', time.time() - s)
    
    predict('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 0', model, False, max_time=2)

    evaluate_model(model)

  if args.play_game:
    model = load_model('model/' + args.play_game[0] + '.h5')

    current_board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    count = 0

    while not current_board.is_game_over():
      print(current_board)
      prediction = predict(current_board.fen(), model, current_board.turn, max_time=2)
      count += 1

      print(str(count) + '. ' + prediction[1])

      if not current_board.is_valid():
        print('Invalid board..')
        break
      if not chess.Move.from_uci(prediction[0]) in current_board.legal_moves:
        print('Invalid move..')
        break

      current_board.push(chess.Move.from_uci(prediction[0]))
    print('Final score: ', current_board.result())

  if args.server:
    current_model = args.server[0]
    app.run(host='0.0.0.0', port=5557)

  # This will not work unless you have the modified version of sunfish
  if args.sunfish:
    current_model = load_model('model/' + args.sunfish[0] + '.h5')
    current_board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    sunfish_board = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
    sunfish_searcher = sunfish.Searcher()
    count = 0

    while not current_board.is_game_over():
      print(current_board)

      if not current_board.is_valid():
        sunfish.print_pos(sunfish_board)
        print('Invalid board..')
        break
      
      prediction = predict(current_board.fen(), current_model, True, max_time=2) # play as white
      if not chess.Move.from_uci(prediction[0]) in current_board.legal_moves:
        print('Invalid move..')
        break

      c = chess.Move.from_uci(prediction[0])
      right_place = chess.square_rank(c.to_square) == 7 or chess.square_rank(c.to_square) == 0
      right_piece = str(current_board.piece_at(c.from_square)) == 'p' or str(current_board.piece_at(c.from_square)) == 'P'
      if right_piece and right_place: 
        print("Promoted")
        c.promotion = 5
      current_board.push(c)

      count += 1


      print('Chazzbot move: ', prediction[0])
      print(current_board)
      
      pred_list = list(prediction[0])
      match = re.match('([a-h][1-8])'*2, prediction[0])
      move = sunfish.parse(match.group(1)), sunfish.parse(match.group(2))
      
      # sunfish make move
      sunfish_board = sunfish_board.move(move)
      sunfish_move, sunfish_score = sunfish_searcher.search(sunfish_board, secs=2)
      sunfish_board = sunfish_board.move(sunfish_move)
      sunfish_move_adjusted = sunfish.render(119-sunfish_move[0]) + sunfish.render(119-sunfish_move[1])

      print('Sunfish move: ', sunfish_move_adjusted)
      print('Sunfish score: ', sunfish_score)
      
      c = chess.Move.from_uci(sunfish_move_adjusted)
      print(chess.square_rank(c.to_square) == 0)
      print(str(current_board.piece_at(c.from_square)) == 'p')
      right_place = chess.square_rank(c.to_square) == 7 or chess.square_rank(c.to_square) == 0
      right_piece = str(current_board.piece_at(c.from_square)) == 'p' or str(current_board.piece_at(c.from_square)) == 'P'
      if right_piece and right_place: 
        print("Promoted")
        c.promotion = 5
      current_board.push(c)
      
      if not current_board.is_valid():
        print('Sunfish broke it...')

      count += 1

    print('Final score (chazzbot-sunfish): ', current_board.result())