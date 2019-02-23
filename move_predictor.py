import keras
import numpy as np
import json
import os
import argparse
import time
import math
import chess
from flask import Flask, request, Response

from data_extractor import convert_fen_label
#from train_network import train_network
from train_network_generator import train_network
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
  fen = str(request.json['fen'])
  
  board = chess.Board(fen + ' b - - 0 0')
  board.turn = False # use fen later
  board.castling_rights = True

  if not c_model:
    c_model = load_model('model/' + current_model + '.h5')

  move = predict(fen + ' b - - 0 0', c_model, False) # look into fen additional params
  
  board.push(chess.Move.from_uci(move[0]))

  res = Response(json.dumps([{"fen": board.fen(), "move": move[0], "explination": move[1]}]),  mimetype='application/json')
  res.headers['Access-Control-Allow-Origin'] = '*'
  return res


#-----------------------------------------
# main
#-----------------------------------------
def predict_depth(board, model, depth=1, t=False):
  '''
  Searches for the best move further down in the search tree
  The depth defines how far the search tree will be searched
  Returns an prediction score.

  note: increasing depth seriously improves performance of 
  estimations but increases the prediction time drastically
  '''
  tmp = 1 if t else 0

  if depth <= 0:
    return 0

  for legal in board.legal_moves:
    board_tmp = board.copy()
    board_tmp.push(legal)
    input_thing = [convert_fen_label(board.fen()) + convert_fen_label(board_tmp.fen())]
    predicted = model.predict(np.array(input_thing))

    predicted = predict_depth(board_tmp, model, depth=depth-1, t=not t)
    
    if (not t and predicted >= tmp) or (t and tmp <= predicted):
      tmp = predicted
    
  return tmp

def predict(fen, model, turn=False):
  '''
  Given keras model and fennotation, and turn: returns the best
  scoring move as well as a descriptive text.
  '''
  board = chess.Board(fen)
  board.turn = turn # use fen later
  tmp = (0, '')

  # For first level save also actual move
  for legal in board.legal_moves:
    board_tmp = chess.Board(fen)
    board_tmp.turn = turn # use fen later
    board_tmp.push(legal)
    
    input_thing = [convert_fen_label(fen) + convert_fen_label(board_tmp.fen())]
    pscore = model.predict(np.array(input_thing))

    # Find accumulated score
    predicted = (pscore + predict_depth(board_tmp, model, depth=2), legal)
    
    print(predicted)
    
    if predicted[0] >= tmp[0]:
      tmp = predicted

  move = tmp[1]
  return str(move), 'Predicted move: ' + str(move) + '. With accumulated prediction being: ' + str(tmp[0][0][0])

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
    print(predict('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1', model, False)[1])
    print(predict('rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2', model, True)[1])
    print(predict('rnbqkbnr/pppp1ppp/8/4p3/4P3/5P2/PPPP2PP/RNBQKBNR b KQkq - 0 2', model, False)[1])
    print(predict('rnbqkbnr/pppp2pp/5p2/4p3/4P3/5P2/PPPP2PP/RNBQKBNR w KQkq - 0 3', model, True)[1])
    print(predict('rnbqkbnr/pppp2pp/5p2/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 3', model, False)[1])

  if args.play_game:
    model = load_model('model/' + args.play_game[0] + '.h5')

    current_board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    count = 0

    while not current_board.is_game_over():
      print(current_board)
      prediction = predict(current_board.fen(), model, current_board.turn)
      print(prediction)

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
