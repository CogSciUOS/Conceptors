
import pickle
from pymongo import MongoClient
from bson.binary import Binary
from datetime import datetime
import inspect
import numpy as np

"""
this weird section of code allows modules in the parent directory to be imported here
it's the only way to do it in a way that allows you to run the file from other directories
and still have it work properly
"""
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import syllableClassifier


'''
This section looks weird and unnecessary. It sorta is but isn't, this allows
the module to act like a singleton (similar to how Python's logging library
works) so that we can use it from multiple files without needing to
reinitialize it or pass it as an argument
'''
logger = None

enabled = True

def configure(db_name):
    if enabled:
        global logger
        logger = MongoLog(db_name)
        logger.clear()

def write(message):
    if enabled:
        if type(message) == str:
            logger.write(message)
        else:
            logger.write_frame_info(message)

def write_val(message, val):
    if enabled:
        logger.write_val(message, val)

def write_arr(message, arr):
    if enabled:
        logger.write_arr(message, arr)

def write_big_arr(message, arr):
    if False:
        logger.write_big_arr(message, arr)

def write_frame_info(frame):
    if enabled:
        logger.write_frame_info(frame)

class MongoLog():

    __slots__ = (
            'conn',
            'coll'
            )

    def __init__(self, db_name):
        self.conn = MongoClient()
        db = self.conn[db_name]
        self.coll = db.log
        # empty out the collection
        self.coll.delete_many({})
        

    def write(self, message):
        self.coll.insert_one({
            "message": message,
            "timestamp": datetime.now()
            })


    def write_val(self, message, val):
        self.coll.insert_one({
            "message": message,
            "timestamp": datetime.now(),
            "value": val
            })


    def write_arr(self, message, arr):
        print(message)
        print(len(arr))
        self.coll.insert_one({
            "message": message,
            "timestamp": datetime.now(),
            "type": "array",
            "value": Binary(pickle.dumps(arr))
            })

    def write_big_arr(self, message, arr):
        print(message)
        print(len(arr))

        total = len(arr) - 1
        for i, row in enumerate(arr):
            self.coll.insert_one({
                'message': message,
                'timestamp': datetime.now(),
                'type': 'multipart array',
                'value': Binary(pickle.dumps(row)),
                'index': i,
                'of': total
                })

    def write_frame_info(self, frame):
        func_name = inspect.getframeinfo(frame).function
        arg_vals = inspect.getargvalues(frame)
        args = {}
        for arg in arg_vals.args:
            val = arg_vals.locals[arg]

            # this is horrible and makes me feel sad :(
            if type(val) == list:
                val = np.array(val)
            if type(val) == np.ndarray:
                # args[arg] = val.to_list()
                args[arg] = 'numpy array'
            elif type(val) == syllableClassifier.syllableClassifier:
                args[arg] = 'syllable classifier'
            else:
                args[arg] = val
        self.coll.insert_one({
            'message': func_name,
            'function': func_name,
            'type': 'frame_info',
            'value': args,
            })


    def clear(self):
        self.coll.remove({})

