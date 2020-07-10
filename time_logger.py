import logging
import time

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class TimeLogger:

    def __init__(self, name):
        self.name = name
        self.startTime = 0
        self.endTime = 0

    def start(self):
        self.startTime = time.time()
        logging.info("Time logger started.")

    def end(self):
        self.endTime = time.time() - self.startTime
        logging.info("Process ended taking " + str(self.endTime / 60.0) + " minutes.")
        return self.endTime
