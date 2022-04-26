import logging
import os
import sys
from datetime import datetime

import pandas as pd
from flask import Flask
from flask_socketio import SocketIO, emit
from numpy import broadcast

from models import Generator, Predictor

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(
                    stream = sys.stdout, 
                    filemode = "w",
                    format = Log_Format, 
                    level = logging.INFO)

logger = logging.getLogger()

###############################################################################
#                               Load models                                   #
###############################################################################
predictor = Predictor('/home/shangling/predictors/')
generator = Generator('/home/shangling/generator')

###############################################################################
#                        Initialize data variables                            #
###############################################################################
# Constants
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
SAVE_PATH = "/home/shangling/test_flask_outputs"
PRED_THRESHOLD = 0.5
DIALOG_COLUMNS = ['user_id', 'is_listener', 'utterance', 'time', 'predictor_input_ids', 'generator_input_ids']
PRED_COLUMNS = ['code', 'score', 'last_utterance_index', 'text']
CLICK_COLUMNS = ['user_id', 'is_listener', 'pred_index', 'time']

# Mutables
client_id, listener_id = "default_client", "default_listener"
dialog_df = pd.DataFrame.from_dict({k: [] for k in DIALOG_COLUMNS})
pred_df = pd.DataFrame.from_dict({k: [] for k in PRED_COLUMNS})
click_df = pd.DataFrame.from_dict({k: [] for k in CLICK_COLUMNS})

def reset_df():
    global dialog_df, pred_df, click_df
    dialog_df = dialog_df[0:0]
    pred_df = pred_df[0:0]
    click_df = click_df[0:0]


###############################################################################
#                      Actually setup the Api resource                        #
###############################################################################
app = Flask(__name__)
socketio = SocketIO(app, logger=logger, engineio_logger=logger, cors_allowed_origins="*")


###############################################################################
#                                    Events                                   #
###############################################################################
@socketio.on("log_user")
def log_user(args):
    """Record who are involved in this conversation

    Args:
        args (json):
            client_id (str): id assigned to client (i.e. us)
            listener_id (str): id assigned to listner (volunteer / human subjects)
    """
    try:
        global client_id, listener_id

        client_id, listener_id = args['client_id'], args['listener_id']
    
    except Exception as e:
        emit("error", str(e))


@socketio.on("add_message")
def add_message(args):
    """Add a new utterance to backend

    Args:
        args (json):
            is_listener (bool): whether the message is sent by the listener
            utterance (str): the new message sent
    
    Emits "new_message"
        is_listener (bool): whether the new message is sent by the listener
        utterance (str): the new message sent
        predictions (List[Tuple[int, str, str]]): list of tuples of (pred_index, MITI code, next utterance)
            The pred_index is used for LogClick described below. 
    """
    try:
        global dialog_df, pred_df, listener_id, client_id

        user_id = listener_id if args["is_listener"] else client_id
        date_time = datetime.now().strftime(DATETIME_FORMAT)
        new_row = [user_id, args["is_listener"], args["utterance"], date_time, [], []]
        last_utterance_index = len(dialog_df.index)
        dialog_df.loc[last_utterance_index] = new_row

        scores = predictor.predict(dialog_df)
        scores = list(filter(lambda x: x[1] > PRED_THRESHOLD, scores))
        scores.sort(key=lambda x: -x[1])
        codes = [x[0] for x in scores]

        generations = generator.predict(dialog_df, codes)

        predictions = []
        for i, (code, utterance) in enumerate(zip(codes, generations)):
            predictions.append((len(pred_df), code, utterance))
            pred_df.at[len(pred_df)] = [code, scores[i][1], last_utterance_index, utterance] 

    except Exception as e:
        emit("error", str(e))

    args["predictions"] = predictions
    emit("new_message", args, broadcast=True)  # Send to all clients


@socketio.on("log_click")
def log_click(args):
    """Record when, who, and what is clicked

    Args:
        args (json):
            is_listener (bool): whether the click is sent by the listener
            pred_index (int): index of the clicked prediction
    """
    try:
        global click_df, listener_id, client_id

        user_id = listener_id if args["is_listener"] else client_id
        date_time = datetime.now().strftime(DATETIME_FORMAT)
        new_row = [user_id, args["is_listener"], args["pred_index"], date_time]
        click_df.loc[len(click_df)] = new_row

    except Exception as e:
        emit("error", str(e))


@socketio.on("dump_logs")
def dump_logs():
    """Store dialog, prediction, and click logs to file and clear the variables
    """
    try:
        global dialog_df, pred_df, click_df, client_id

        now = datetime.now()
        date_time = now.strftime(DATETIME_FORMAT)
        prefix = f"{SAVE_PATH}/{client_id}_{date_time}_"

        dialog_df.to_csv(prefix + "dialog.csv", index=False)
        pred_df.to_csv(prefix + "pred.csv", index=False)
        click_df.to_csv(prefix + "click.csv", index=False)

        reset_df()

    except Exception as e:
        emit("error", str(e))


if __name__ == '__main__':
    socketio.run(app, debug=False)
