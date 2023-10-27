from signalrcore.hub_connection_builder import HubConnectionBuilder
import pytest
import time
import logging
from collections import defaultdict
from datetime import datetime

# create logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("GSConnector")



class GSConnector:
    def __init__(self, target_language="cpp"):
        self.target_language = target_language
    
    def setup_hubconnection(self):
        self.callbacks = defaultdict(list)
        hub_connection = (
            HubConnectionBuilder()
            .with_url("https://localhost:5050/frontendhub", options={"verify_ssl": False})
            .configure_logging(logging.INFO)
            .with_automatic_reconnect(
                {
                    "type": "raw",
                    "keep_alive_interval": 10,
                    "reconnect_interval": 5,
                    "max_attempts": 5,
                }
            )
            .build()
        )

        # Define a handler for the on_connect event
        def on_connect():
            print(f"Test program to: Connection started")

        # Define a handler for the on_reconnect event
        def on_reconnect():
            print(f"Test program to: Connection restablished")

        # Define a handler for the on_disconnect event
        def on_disconnect():
            print(f"Test program to: Connection closed")

        # Define a handler for the message event
        def on_message(msg):
            # print(f"Test program to: Message received: {msg}")
            pass

        def on_log_terminal(msg):
            print(f"Test program to: Log terminal: {msg}")
            # msg can be of type list[str] or list[str, str]
            if len(msg) == 2:
                # we got a message with error
                if "error" in msg[0].lower():
                    self.callbacks["log_error"].append(msg[1])
            # save as list[str]
            self.callbacks["log_terminal"].append(msg[0])

        def on_log_solution(msg):
            print(f"Test program to: Log solution: {msg}")
            self.callbacks["log_solution"].append(msg)

        def on_log_ut(msg):
            print(f"Test program to: Log UT: {msg}")
            self.callbacks["log_ut"].append(msg)

        def on_log_translation(msg):
            print(f"Test program to: Log T: {msg}")
            self.callbacks["log_translation"].append(msg)

        # register callbacks logTerminal, logSolution, logUT
        hub_connection.on_open(on_connect)
        hub_connection.on_close(on_disconnect)
        hub_connection.on_reconnect(on_reconnect)
        hub_connection.on("logTerminal", on_log_terminal)
        hub_connection.on("logSolution", on_log_solution)
        hub_connection.on("logUT", on_log_ut)
        hub_connection.on("logT", on_log_translation)
        hub_connection.on("message", on_message)

        hub_connection.start()
        # wait 1 sec for connection to establish
        time.sleep(1)
        self.hub_connection = hub_connection
    

    def send_convert_request(self, code):
        self.setup_hubconnection()
        self.callbacks = defaultdict(list)
        self.hub_connection.send("ConvertAsync", [code, self.target_language])

        # wait until solution is received
        max_wait = 120
        start_time = time.time()
        while len(self.callbacks["log_solution"]) < 1:
            if time.time() - start_time > max_wait:
                print("Timeout")
                break
            time.sleep(2)

        # Stop the connection
        self.hub_connection.stop()

        # take solution or first translation
        solution = [""]
        if len(self.callbacks["log_solution"]) > 0 and len(self.callbacks["log_translation"][0]) > 0 and self.callbacks["log_solution"][0][0] != "":
            solution = self.callbacks["log_solution"][0]
        elif len(self.callbacks["log_translation"]) > 0:
            solution = self.callbacks["log_translation"][0]
        else:
            logger.error("No solution or translation received")
        return solution