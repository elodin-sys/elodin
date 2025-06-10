#!/usr/bin/env python3

import numpy as np
import elodin_db
import time


def main():
    # connects to elodin db
    client = elodin_db.ElodinDB.start()
    print(client.addr)
    client.set_entity_metadata(2, "sensor")

    # send a world pos as a table
    print("Sending position data...")
    position_data = np.array([0, 0, 0, 1.0, 1.0, 2.0, 3.0])  # q0,q1,q2,q3, x, y, z coordinates
    client.send_table(entity_id=1, component_id="world_pos", data=position_data)

    print("sending sensor data...")
    
    client.set_component_metadata("temps")
    t = 0
    while True:
        noise = np.random.rand(3)
        t += 1.0 / 500.0
        reading = np.array([np.sin(t * 2 * 3.14 / 100) + np.sin(t * 2 * 3.14 / 20), np.cos(t), np.sin(t * 2 * 3.14 / 200) + np.sin(t * 2 * 3.14 / 10)]) + noise
        client.send_table(entity_id=2, component_id="temps", data=reading)
        time.sleep(1.0 / 500)        


if __name__ == "__main__":
    main()
