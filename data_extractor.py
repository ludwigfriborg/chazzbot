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
  'P': -1,
  'R': -2,
  'N': -3,
  'B': -4,
  'Q': -5,
  'K': -6
}
char_dict_b = {
  'p': -1,
  'r': -2,
  'n': -3,
  'b': -4,
  'q': -5,
  'k': -6,
  'P': 1,
  'R': 2,
  'N': 3,
  'B': 4,
  'Q': 5,
  'K': 6
}


def reshape_moves(board, move):
  return move.tolist()
  #return np.concatenate(move).tolist()
  #return np.concatenate(board).tolist() + np.concatenate(move).tolist()
  #return np.concatenate([board, move])

def convert_fen_label(fen, flip):
  parts = fen.split(' ')
  if flip:
    board = fill_fen_board(parts[0], False if parts[1] == 'b' else True)
  else:
    board = fill_fen_board(parts[0], False if parts[1] == 'w' else True)
    
  return indivualize_board(board)

def fill_fen_board(b, flip):
  rows = b.split('/')
  if flip:
    rows.reverse()
  letters = []
  for row in rows:
    for char in list(row):
      if char.isalpha():
        if not flip:
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
      board_indivualized[-piece - 1][i] = -1
    else:
      board_indivualized[piece-1][i] = 1
  return np.concatenate(board_indivualized)

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

    #if whowon == '1/2-1/2':
    #  continue

    board = game.board()

    if progress % 10 == 0:
      print('{0}/{1} <{2}> Number of games analyzed: {3}/{4} ({5}%)'.format(this_file, num_files, file_name, progress, num_of_games, int(100*progress/num_of_games)), end='\r')
    progress += 1
    for move in game.main_line():
      tmp_board=board.copy()

      if whowon == '1-0' and tmp_board.turn != True:
        continue
      elif whowon == '0-1' and tmp_board.turn == True:
        continue
      # at the moment only keep winning games
      tmp_b = convert_fen_label(str(board.fen()), False)
      board.push(move)
      move_to_save = reshape_moves(tmp_b, convert_fen_label(str(board.fen()), True))
      move_to_save.append(1) # this is winning move
    
      data.append(move_to_save)

      #random move
      #for generating winning predictor
      count = round(tmp_board.pseudo_legal_moves.count() * random.randint(1,101) / 100)
      for m in tmp_board.pseudo_legal_moves:
        move = m
        count -= 1
        if count < 0:
          break

      tmp_board.push(move)
      move_to_save_n = reshape_moves(tmp_b, convert_fen_label(str(tmp_board.fen()), True))
      move_to_save_n.append(0) # this is loosing move
      
      data.append(move_to_save_n)

  print('<{0}> Number of games analyzed: {1}/{2} (100%)'.format(file_name, progress, num_of_games))
  return data


if __name__ == "__main__":
  file_names = []
  for file in os.listdir("data/pgn_a_l"):
      if file.endswith(".pgn"):
        file_names.append(file.split('.pgn')[0])

  success_count = 0
  all_data = []
  chunk_move = 40000 #how many moves per file
  setname = 'value'
  num_files = math.floor(len(file_names))
  skip_count = 0 # how many files in to skip
  index_num = 0 # what name index generation should start on
  
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
      if success_count > num_files:
        break

    except Exception as e:
      print('File failed: ', file_n)
      print(e)

  print('Successfully loaded: {0}/{1}'.format(success_count, len(file_names)))