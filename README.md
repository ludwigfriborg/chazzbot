# Chazzbot
Chess ai, using feedforward neural net to assess value to proposed moves. [more info here about how the network works]

Host flask server, can be played against at: schack.ludwigfriborg.se. Something about the custom web application.

## Manual
- `python3 data_extractor.py`, on run converts all pgn data contained in the folder `data` to the models internal format. Stores the converted data in the folder `ext` as json.
- `python3 move_predictor.py -t <name>`, trains a new model with the data in `ext` and stores the model in `model` under the given model name
- `python3 move_predictor.py -pg <name>`, initiates a game for the ai to play against it self. Uses the model with the name given as argument.
- `python3 move_predictor.py -s <name>`, launches as simple rest api for clients to requst predictions. Uses the model with the name given as argument.

[something about the uploaded model]