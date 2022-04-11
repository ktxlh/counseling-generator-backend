from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from models import Predictor, Generator
import pandas as pd
from datetime import datetime

SAVE_PATH = "/data/shsu70/test_flask_outputs"
PRED_THRESHOLD = 0.5

###############################################################################
#                               Load models                                   #
###############################################################################
predictor = Predictor('/data/shsu70/testing/predictors/')  # TODO change to predictors
generator = Generator('/data/shsu70/models/microsoft-DialoGPT-small-finetuned-7cups-321606/checkpoint-5200')  # TODO change model path?


###############################################################################
#                        Initialize data variables                            #
###############################################################################
client_id, listener_id = "default_client", "default_listener"

dialog_columns = ['user_id', 'is_listener', 'utterance', 'time', 'predictor_input_ids', 'generator_input_ids']
pred_columns = ['code', 'score', 'last_utterance_index', 'text']
click_columns = ['user_id', 'is_listener', 'pred_index', 'time']

def reset_df():
    return [
        pd.DataFrame.from_dict({k: [] for k in columns}) 
        for columns in [dialog_columns, pred_columns, click_columns]
    ]

dialog_df, pred_df, click_df = reset_df()


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
        args = user_parser.parse_args()
        client_id, listener_id = args['client_id'], args['listener_id']


message_parser = reqparse.RequestParser()
message_parser.add_argument('is_listener', type=bool, help="whether the message is sent by the listener")
message_parser.add_argument('utterance', type=str, help="the new message sent")
message_parser.add_argument('time', type=str, help="time sent")
class AddMessage(Resource):
    def post(self):
        """Add a new utterance to backend
        """
        args = message_parser.parse_args()
        user_id = listener_id if args["is_listener"] else client_id
        new_row = [user_id, args["is_listener"], args["utterancee"], args["time"], [], []]
        dialog_df.loc[len(dialog_df.index)] = new_row

        scores = predictor.predict(dialog_df)
        scores = list(filter(lambda x: x[1] > PRED_THRESHOLD, scores))
        scores.sort(key=lambda x: -x[1])
        codes = [x[0] for x in scores]

        generations = generator.predict(dialog_df, codes)



click_parser = reqparse.RequestParser()
click_parser.add_argument('is_listener', type=bool, help="whether the click is sent by the listener")
click_parser.add_argument('pred_index', type=int, help="index of the clicked prediction")
click_parser.add_argument('time', type=str, help="time clicked")
class LogClick(Resource):
    def post(self):
        """Record when, who, and what is clicked
        """
        args = click_parser.parse_args()
        user_id = listener_id if args["is_listener"] else client_id
        new_row = [user_id, args["is_listener"], args["pred_index"], args["time"]]
        click_df.loc[len(click_df)] = new_row
        return 200


class DumpLogs(Resource):
    def get(self):
        """Store dialog, prediction, and click logs to file and clear the variables
        """
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
        prefix = f"{SAVE_PATH}/{client_id}_{date_time}_"
        try:
            dialog_df.to_csv(prefix + "dialog.csv")
            pred_df.to_csv(prefix + "pred.csv")
            click_df.to_csv(prefix + "click.csv")
            reset_df()
        except Exception as e:
            return e, 500
        return 200


###############################################################################
#                                    Main                                     #
###############################################################################
app = Flask(__name__)
api = Api(app)
if __name__ == '__main__':
    app.run(debug=True)
