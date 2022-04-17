# Counseling Generator Backend

## Setup
### Install Miniconda

Miniconda installer
This [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) contains all comments I use with conda.

After installing miniconda, create an env `mi`.
```
conda env create --name mi python=3.9
```

Run this to activate the env every time you work on this project.
```
conda activate mi
```

For the first time you use the env, run this inside (`counseling-generator-backend/`) to install the dependencies.
```
pip install -r requirements.txt
```

After you installed anything new, please add it to `requirements.txt` manually following the same format.

## Backend APIs
1. Download the models from [Google Drive](https://drive.google.com/drive/folders/1nfpSg6q6meDs3JuKcFMrk3COYxxtyBwh?usp=sharing).
2. In [api.py](https://github.com/ktxlh/counseling-generator-backend/blob/main/api.py), Replace the 3 paths containing '/data/shsu70' with your config (2 of them should match the model paths in 1.)
3. Run backend:
```
python api.py
```

Send all requests over http
* All `POST` requests send arguments in **body** in **raw json** format and return values in **json** format.
* The following demo the API at the default Flask port `http://localhost:5000`, but it can be hosted on any server address and port.
* Server returns 200 when everything goes well, 500 when not. See the list of error codes [here](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status)
* Postman collections I used for testing: [![Run in Postman](https://run.pstmn.io/button.svg)](https://app.getpostman.com/run-collection/c50bfe23ff6a87e0472a)

### Log User
* Change the value of listener's and client's ids from the backend. This should be called before each session starts.
* Method: `POST`
* URL: http://localhost:5000/loguser
* Example arguments:
```
{
    "client_id": "client-id-test",
    "listener_id": "listener-id-test"
}
```

### Add Message
* Whenever any user sends a message, run this to get predictions and generations, and record it on server. 
Each generation comes with a prediction index. It will be used for the next API (Log Click) so keep it at frontend when the predictions are still clickable (see below).
Note that when the dialog history isn't long enough (i.e. <5), it returns an empty list.
* Method: `POST`
* URL: http://localhost:5000/addmessage
* Example arguments:
```
{
    "is_listener": true,
    "utterance": "How are you doing?"
}
```
* Example return values:
```
[
    [
        0,
        "PR",
        "good, i'm bad at these"
    ],
    [
        1,
        "QUC",
        "good u."
    ],
    [
        2,
        "RF",
        "How How??"
    ],
    [
        3,
        "GR",
        "Good."
    ],
    [
        4,
        "SUP",
        "good u im trying ahhh"
    ],
    [
        5,
        "AF",
        "how u doing"
    ]
]
```

### Log Click
* Whenever the user clicks a generation, run this to record it on server.
* Method: `POST`
* URL: http://localhost:5000/logclick
* Example arguments:
```
{
    "is_listener": true,
    "pred_index": 13
}
```

### Dump Logs
* Save all logs to files on server and clear them (including the dialog!). This should be run after each session.
* Method: `GET`
* URL: localhost:5000/dumplogs
* No arguments!
