# ![](frontend/img/chazzbot.jpeg)Chazzbot
Chess AI, using a feed-forward neural network as a value function and some simple minimax tree search. Build with Keras and Tensorflow trained on Grandmaster championship games fetched from [https://www.pgnmentor.com/files.html.](https://www.pgnmentor.com/files.html) 

The project uses a flask server as an interface and python-chess up to v.0.23.x for chess related stuff.

The project now supports playing games against the [Sunfish](<https://github.com/thomasahle/sunfish>).

*The project is using Tensorflow 1.13.1 and Keras 2.2.4*

## Manual
- `python3 data_extractor.py`, on run converts all `.pgn` data contained in the folder `data` to the model's internal format. Stores the converted data in the folder `ext` as json.
- `python3 move_predictor.py -t <name>`, trains a new model with the data in `ext` and stores the model in `model` under the given model name
- `python3 move_predictor.py -pg <name>`, initiates a game for the AI to play against itself. Uses the model with the name given as argument.
- `python3 move_predictor.py -s <name>`, launches as simple rest api for clients to request predictions. Uses the model with the name given as argument.
- `python3 move_predictor.py -sun <name>`, starts a game against the ai Sunfish which have been included into this project.

A working model has been added to the project called `model` and can simply be used to run the application. The uploaded setup searches for 2 seconds by default, though this can be changed.

## Potential improvements

- [x] Apply some search algorithm to find the best move quicker.
- [x] Add some alpha-beta pruning.
- [x] Add support for castling and so on for the back-end.
- [x] Improve and optimize the minimax search
- [ ] Potentially implement Monte Carlo tree search
- [ ] Improve the neural network accuracy
- [ ] Add support for python-chess 2.5+
