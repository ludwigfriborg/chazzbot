import keras
import numpy as np
import json
import os
import argparse
import time
import math
import chess
from flask import Flask, request, Response

from data_extractor import convert_fen_label, reshape_moves
#from train_network import train_network
from train_network_generator import train_network, evaluate_model
from keras.models import Sequential, load_model

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)
current_model = ''
c_model = False

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

  if chess.Move.from_uci(move[0]) in board.pseudo_legal_moves or board.is_castling(chess.Move.from_uci(move[0])):
    board.push(chess.Move.from_uci(move[0]))

    if board.is_checkmate():
      move = (move[0], 'Check mate - you win')
    else:
      if not c_model:
        c_model = load_model('model/' + current_model + '.h5')

      move = predict(board.fen(), c_model, False) # look into fen additional params
      board.push(chess.Move.from_uci(move[0]))
      if board.is_checkmate():
        move = (move[0], 'Check mate - chazzbot wins')

  res = Response(json.dumps([{"fen": board.fen(), "move": move[0], "explination": move[1]}]),  mimetype='application/json')
  res.headers['Access-Control-Allow-Origin'] = '*'
  return res


#-----------------------------------------
# main
#-----------------------------------------
def scale_pred(pred):
  if pred < 0.25:
    return 1
  if pred < 0.5:
    return 2  
  if pred < 0.75:
    return 4
  return 10

def predict_depth(score, board, model, depth=1, minimizing=True, a_i=math.inf, b_i=-math.inf):
  '''
  Searches for the best move further down in the search tree
  The depth defines how far the search tree will be searched
  Returns an prediction score.

  note: increasing depth seriously improves performance of 
  estimations but increases the prediction time drastically
  '''
  tmp = math.inf if minimizing else -math.inf
  a, b = a_i, b_i

  if board.is_checkmate():
    # The punishments are reversed here since the check happens on the next layer
    bingo = score - 1000 if not minimizing else score + depth * 10
    return bingo

  if depth <= 0:
    return score

  for legal in board.legal_moves:
    board_tmp = board.copy()
    board_tmp.push(legal)
    input_thing = reshape_moves(convert_fen_label(board.fen()), convert_fen_label(board_tmp.fen()))
    input_thing = np.array([input_thing])
    pred = model.predict(input_thing)

    if (not minimizing):
      pscore = score + scale_pred(pred)
      predicted = predict_depth(pscore, board_tmp, model, depth=depth-1, minimizing=True, a_i=a, b_i=b)

      tmp = max(tmp, predicted)
      a = max(a, tmp)
      if b <= a:
        return tmp

    if (minimizing):
      pscore = score - scale_pred(pred)
      predicted = predict_depth(pscore, board_tmp, model, depth=depth-1, minimizing=False, a_i=a, b_i=b)

      tmp = min(tmp, predicted)
      b = min(b, tmp)
      if b <= a:
        return tmp
        
  return tmp

def predict(fen, model, turn=False):
  '''
  Given keras model and fennotation, and turn: returns the best
  scoring move as well as a descriptive text.
  '''
  board = chess.Board(fen)
  board.turn = turn # use fen later
  tmp = (-math.inf, '')
  max_depth = 16

  # For first level save also actual move
  for legal in board.legal_moves:
    board_tmp = board.copy()
    board_tmp.push(legal)
    
    input_thing = reshape_moves(convert_fen_label(fen), convert_fen_label(board_tmp.fen()))
    input_thing = np.array([input_thing])
    pscore = scale_pred(model.predict(input_thing))

    # Find accumulated score
    predicted = (predict_depth(pscore, board_tmp, model, depth=max_depth), legal)
    
    print(predicted)
    if tmp[0] <= predicted[0]:
      tmp = predicted

  move = tmp[1]
  return str(move), 'Predicted move: ' + str(move) + '. With accumulated prediction being: ' + str(tmp[0])

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Chezzbot, chess move predictor using a regular feed forward network!')
  parser.add_argument('-t','--train', help='Trains model', nargs=1)
  parser.add_argument('-st','--standard-test', help='Tests model on some standard moves', nargs=1)
  parser.add_argument('-pg','--play-game', help='plays quick game', nargs=1)
  parser.add_argument('-s','--server', help='start flask service', nargs=1)

  args = parser.parse_args()

  if args.train:
    train_network(args.train[0])

  if args.standard_test:
    model = load_model('model/' + args.standard_test[0] + '.h5')
    evaluate_model(model)

  if args.play_game:
    model = load_model('model/' + args.play_game[0] + '.h5')

    current_board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    count = 0

    while not current_board.is_game_over():
      print(current_board)
      prediction = predict(current_board.fen(), model, current_board.turn)
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
