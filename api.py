import os
from datetime import datetime

import pandas as pd
from flask import Flask
from flask_restful import Api, Resource, reqparse

from models import Generator, Predictor

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

###############################################################################
#                               Load models                                   #
###############################################################################
predictor = Predictor('/data/shsu70/testing/predictors/')  # TODO change to predictors
generator = Generator('/data/shsu70/testing/generator')  # TODO change to generator


###############################################################################
#                        Initialize data variables                            #
###############################################################################
# Constants
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
SAVE_PATH = "/data/shsu70/test_flask_outputs"
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
#                                  Resources                                  #
###############################################################################
user_parser = reqparse.RequestParser()
user_parser.add_argument('client_id', type=str, help="id assigned to client (i.e. us)")
user_parser.add_argument('listener_id', type=str, help="id assigned to listner (volunteer / human subjects)")
class LogUser(Resource):
    def post(self):
        """Record who are involved in this conversation
        """
        try:
            global client_id, listener_id

            args = user_parser.parse_args()
            client_id, listener_id = args['client_id'], args['listener_id']
        
        except Exception as e:
            return str(e), 500

        return 200


message_parser = reqparse.RequestParser()
message_parser.add_argument('is_listener', type=bool, help="whether the message is sent by the listener")
message_parser.add_argument('utterance', type=str, help="the new message sent")
class AddMessage(Resource):
    def post(self):
        """Add a new utterance to backend

        Returns:
            predictions (List[Tuple[int, str, str]]): list of tuples of (pred_index, MITI code, next utterance)
                The pred_index is used for LogClick described below. 
        """
        try:
            global message_parser, dialog_df, pred_df, listener_id, client_id

            args = message_parser.parse_args()
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
            return str(e), 500

        return predictions, 200



click_parser = reqparse.RequestParser()
click_parser.add_argument('is_listener', type=bool, help="whether the click is sent by the listener")
click_parser.add_argument('pred_index', type=int, help="index of the clicked prediction")
class LogClick(Resource):
    def post(self):
        """Record when, who, and what is clicked
        """
        try:
            global click_parser, click_df, listener_id, client_id

            args = click_parser.parse_args()
            user_id = listener_id if args["is_listener"] else client_id
            date_time = datetime.now().strftime(DATETIME_FORMAT)
            new_row = [user_id, args["is_listener"], args["pred_index"], date_time]
            click_df.loc[len(click_df)] = new_row

        except Exception as e:
            return str(e), 500

        return 200


class DumpLogs(Resource):
    def get(self):
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
            return str(e), 500

        return 200


###############################################################################
#                      Actually setup the Api resource                        #
###############################################################################
app = Flask(__name__)
api = Api(app)
api.add_resource(LogUser, '/loguser')
api.add_resource(AddMessage, '/addmessage')
api.add_resource(LogClick, '/logclick')
api.add_resource(DumpLogs, '/dumplogs')


if __name__ == '__main__':
    app.run(debug=True)
