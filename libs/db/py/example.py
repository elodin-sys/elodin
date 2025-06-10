#!/usr/bin/env python3

import numpy as np
import elodin_db
import time


def main():
    # connects to elodin db
    client = elodin_db.ElodinDB.start()
    print(client.addr)

    # send a world pos as a table
    print("Sending position data...")
    position_data = np.array([0, 0, 0, 1.0, 1.0, 2.0, 3.0])  # q0,q1,q2,q3, x, y, z coordinates
    client.send_table(entity_id=1, component_id="world_pos", data=position_data)

    print("sending sensor data...")
    
    while True:
        reading = np.random.rand(3)
        client.send_table(entity_id=2, component_id="temp", data=reading)
        time.sleep(1.0 / 500)        


if __name__ == "__main__":
    main()
