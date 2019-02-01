import chess
import chess.pgn
import json
from io import StringIO
import os
import random

def split(arr, size):
    arrs = []
    while len(arr) > size:
      pice = arr[:size]
      arrs.append(pice)
      arr   = arr[size:]
    arrs.append(arr)
    return arrs

# data notation
# { board: labels, fen: FEn board, move: labels, winning: binary }

char_dict = {
  'p': 1,
  'r': 2,
  'n': 3,
  'b': 4,
  'q': 5,
  'k': 6,
  'b': 7,
  'P': 8,
  'R': 9,
  'N': 10,
  'B': 11,
  'Q': 12,
  'K': 13,
  'B': 14,
}


def convert_fen_label(fen):
  parts = fen.split(' ')
  board = fill_fen_board(parts[0])
  # return board as array
  # black or white
  board.append(1 if parts[1] == 'w' else 0)

  return board

def fill_fen_board(b):
  rows = b.split('/')
  letters = []
  for row in rows:
    for char in list(row):
      if char.isalpha():
        #letters.append(char_dict[char]/len(char_dict)) # normaize
        letters.append(char_dict[char])
      else:
        [letters.append(0) for x in range(0, int(char))]

  return letters

def get_training_data(file_name):
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
    board = game.board()

    if progress % 10 == 0:
      print('<{0}> Number of games analyzed: {1}/{2} ({3}%)'.format(file_name, progress, num_of_games, int(100*progress/num_of_games)), end='\r')
    progress += 1
    for move in game.main_line():
      tmp_board=board

      item = {}
      item['board'] = convert_fen_label(str(board.fen()))
      item['fen'] = str(board.fen())
      item['turn'] = 1 if board.turn else 0

      # make move and save it as board gives move
      board.push(move)
      move = board.peek()
      item['move'] = convert_fen_label(str(board.fen()))

      item['winning'] = 1
      data.append(item)

      #random move
      #for generating winning predictor
      count = round(tmp_board.pseudo_legal_moves.count() * random.randint(1,101) / 100)
      for m in tmp_board.pseudo_legal_moves:
        move = m
        count -= 1
        if count < 0:
          break

      item1 = {}
      item1['turn'] = 1 if tmp_board.turn else 0
      tmp_board.push(move)
      move = tmp_board.peek()

      item1['board'] = convert_fen_label(str(board.fen()))
      item1['fen'] = str(board.fen())
      item1['move'] = convert_fen_label(str(board.fen()))[:65]
      item1['winning'] = 0
      data.append(item1)

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

  for file_n in file_names:
    try:
      all_data = get_training_data(file_n)

      for i in range(round(len(all_data)/chunk_move)):
        index_num += 1
        with open('ext/extracted_data_'+ setname + '_' + str(index_num) + '.json', 'w') as outfile:
          json.dump(all_data[(i)*chunk_move:(i)*chunk_move+chunk_move], outfile)

      all_data = []
      success_count += 1
      if success_count > (len(file_names)/2):
        break

    except Exception as e:
      print('File failed: ', file_n)
      print(e)

  print('Successfully loaded: {0}/{1}'.format(success_count, len(file_names)))