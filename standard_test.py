import time
import numpy as np
from data_extractor import convert_fen_label, reshape_moves
from train_network_generator import evaluate_model


def standard_test(model, prediction_function):
    print(model.summary())

    input_thing = reshape_moves(convert_fen_label('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 0', False), convert_fen_label('rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w - - 0 0', True))
    input_thing = np.array([input_thing])
    s = time.time()
    model.predict(input_thing)[0][0]
    print('Time it took (in s):', time.time() - s)
    
    prediction_function('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 0', False, max_time=1)

    evaluate_model(model)