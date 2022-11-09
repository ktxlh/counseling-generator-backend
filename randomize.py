import pandas as pd
import re
from random import random

ID_PATH = "/home/ubuntu/models/chat_user_ids.csv"

df = pd.read_csv(ID_PATH)

ex = "^[a-zA-Z]{3}$"
mask = [re.search(ex, x) != None for x in df.chat_id]

cols = ["user_id", "chat_id_1", "chat_id_2", "show_suggestion_first"]
new_df = pd.DataFrame.from_dict({k: [] for k in cols})

user_ids = list(set(df[mask].user_id))
user_ids.sort()

for user_id in user_ids:
    rows = df[df.user_id == user_id]
    show_suggestion_first = random() > 0.5
    if show_suggestion_first:
        new_df.loc[len(new_df)] = [user_id, rows.chat_id.tolist()[0], rows.chat_id.tolist()[1], True]
    else:
        new_df.loc[len(new_df)] = [user_id, rows.chat_id.tolist()[1], rows.chat_id.tolist()[0], False]

new_df.to_csv("/home/ubuntu/models/participant_ids.csv", index=False, sep="\t")
