#!/usr/bin/python3
import chess
import chess.pgn
import json
from io import StringIO
import os
import random
import numpy as np
import math

def split(arr, size):
    arrs = []
    while len(arr) > size:
      pice = arr[:size]
      arrs.append(pice)
      arr   = arr[size:]
    arrs.append(arr)
    return arrs

char_dict_w = {
  'p': 1,
  'r': 2,
  'n': 3,
  'b': 4,
  'q': 5,
  'k': 6,
  'b': 7,
  'P': -1,
  'R': -2,
  'N': -3,
  'B': -4,
  'Q': -5,
  'K': -6,
  'B': -7,
}
char_dict_b = {
  'p': -1,
  'r': -2,
  'n': -3,
  'b': -4,
  'q': -5,
  'k': -6,
  'b': -7,
  'P': 1,
  'R': 2,
  'N': 3,
  'B': 4,
  'Q': 5,
  'K': 6,
  'B': 7,
}


def reshape_moves(board, move):
  return np.concatenate(board).tolist() + np.concatenate(move).tolist()

def convert_fen_label(fen):
  parts = fen.split(' ')
  board = fill_fen_board(parts[0], 1 if parts[1] == 'w' else 0)

  return indivualize_board(board)

def fill_fen_board(b, t):
  rows = b.split('/')
  if not t:
    rows.reverse()
  letters = []
  for row in rows:
    for char in list(row):
      if char.isalpha():
        if t:
          letters.append(char_dict_w[char])
        else:
          letters.append(char_dict_b[char])
      else:
        [letters.append(0) for x in range(0, int(char))]

  return letters

# individual labels
def indivualize_board(board):
  board_indivualized = np.zeros((7, 64), dtype=int)
  for i, piece in enumerate(board):
    if piece == 0:
      continue
    if piece < 0:
      board_indivualized[-1*piece-1][i] = -1
    else:
      board_indivualized[piece-1][i] = 1
  return board_indivualized.tolist()

def get_training_data(file_name, num_files=0, this_file=0):
  file = open("data/pgn_a_l/" + file_name + ".pgn").read()

  #load games
  games_as_strings = file.split('[Event')
  num_of_games = len(games_as_strings)

  del games_as_strings[0]

  data = []
  progress = 0

  #load seperate game
  for game_as_string in games_as_strings:
    pgn = StringIO('[Event' + game_as_string)
    game = chess.pgn.read_game(pgn)
    whowon = game.headers['Result']
    if whowon == '1/2-1/2':
      continue

    board = game.board()

    if progress % 10 == 0:
      print('{0}/{1} <{2}> Number of games analyzed: {3}/{4} ({5}%)'.format(this_file, num_files, file_name, progress, num_of_games, int(100*progress/num_of_games)), end='\r')
    progress += 1
    for move in game.main_line():
      tmp_board=board.copy()

      #print(whowon)
      win = 1
      if whowon == '1-0' and tmp_board.turn != True:
        #continue
        win = 0
      elif whowon == '0-1' and tmp_board.turn == True:
        #continue
        win = 0
      if win == 1:
        tmp_b = convert_fen_label(str(board.fen()))
        board.push(move)
        move = reshape_moves(tmp_b, convert_fen_label(str(board.fen())))
        move.append(win) # this is winning move
      

        data.append(move)

      #random move
      #for generating winning predictor
      count = round(tmp_board.pseudo_legal_moves.count() * random.randint(1,101) / 100)
      for m in tmp_board.pseudo_legal_moves:
        move = m
        count -= 1
        if count < 0:
          break

      tmp_b1 = convert_fen_label(str(tmp_board.fen()))
      tmp_board.push(move)
      move1 = reshape_moves(tmp_b1, convert_fen_label(str(tmp_board.fen())))
      move1.append(0) # this is loosing move
      data.append(move1)

  print('<{0}> Number of games analyzed: {1}/{2} (100%)'.format(file_name, progress, num_of_games))
  return data


if __name__ == "__main__":
  file_names = []
  for file in os.listdir("data/pgn_a_l"):
      if file.endswith(".pgn"):
        file_names.append(file.split('.pgn')[0])

  success_count = 0
  all_data = []
  chunk_move = 50000 #how many moves per file
  index_num = 0
  setname = 'value'
  num_files = len(file_names)
  skip_count = 0

  for file_index, file_n in enumerate(file_names):
    # skips as many input files.
    if skip_count > 0:
      skip_count -= 1
      continue
    try:
      all_data = get_training_data(file_n, num_files=num_files, this_file=file_index+1)
      print(len(all_data))
      for i in range(math.ceil(len(all_data)/chunk_move)):
        index_num += 1
        with open('ext/extracted_data_'+ setname + '_' + str(index_num) + '.json', 'w') as outfile:
          json.dump(all_data[(i)*chunk_move:(i)*chunk_move+chunk_move], outfile)

      all_data = []
      success_count += 1
      if success_count > (len(file_names)):
        break

    except Exception as e:
      print('File failed: ', file_n)
      print(e)

  print('Successfully loaded: {0}/{1}'.format(success_count, len(file_names)))