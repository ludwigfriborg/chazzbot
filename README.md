# ![](frontend/img/chazzbot.jpeg)Chazzbot
Chess AI, using a feed-forward neural network as a value function and some simple minimax tree search. Build with Keras and Tensorflow trained on Grandmaster championship games fetched from [https://www.pgnmentor.com/files.html.](https://www.pgnmentor.com/files.html) 

The project uses a flask server as an interface and python-chess for chess related stuff.

## Manual
- `python3 data_extractor.py`, on run converts all `.pgn` data contained in the folder `data` to the model's internal format. Stores the converted data in the folder `ext` as json.
- `python3 move_predictor.py -t <name>`, trains a new model with the data in `ext` and stores the model in `model` under the given model name
- `python3 move_predictor.py -pg <name>`, initiates a game for the AI to play against itself. Uses the model with the name given as argument.
- `python3 move_predictor.py -s <name>`, launches as simple rest api for clients to request predictions. Uses the model with the name given as argument.

A working model can be downloaded [here](https://www.dropbox.com/s/h4tez83dd7hreuq/model_old.h5?dl=0). It should be placed in a directory with the name "model" in the project root. The uploaded setup searches only 4 moves forwards by default, though this can be changed.

## Potential improvements

- [x] Apply some search algorithm to find the best move quicker.

- [x] Add some alpha-beta pruning.

- [x] Add support for castling and so on for the back-end.

- [ ] Improve and optimize the minimax search
- [ ] Improve the neural network accuracy
