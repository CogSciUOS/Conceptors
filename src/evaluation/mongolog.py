
import pickle
from pymongo import MongoClient
from bson.binary import Binary
from datetime import datetime


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
        logger.write(message)

def write_val(message, val):
    if enabled:
        logger.write_val(message, val)

def write_arr(message, arr):
    if enabled:
        logger.write_arr(message, arr)

def write_big_arr(message, arr):
    if enabled:
        logger.write_big_arr(message, arr)

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

    def clear(self):
        self.coll.remove({})

