#!/usr/bin/env python3

import numpy as np
import elodin


def main():
    # connects to elodin db
    client = elodin.ElodinClient("127.0.0.1:2240")

    # send a world pos as a table
    print("Sending position data...")
    position_data = np.array([0, 0, 0, 0, 1.0, 1.0, 2.0, 3.0])  # q0,q1,q2,q3, x, y, z coordinates
    client.send_table(entity_id=1, component_id="world_pos", data=position_data)

    print("sending sensor data...")
    sensor_readings = np.linspace(0.0, 100.0, 50)  # 50 sensor readings
    client.send_table(entity_id=2, component_id="temp", data=sensor_readings)

    print("sending time series data...")
    timestamps = np.arange(0.0, 10.0, 0.1)  # 100 time points
    client.send_table(entity_id=3, component_id="velocity", data=timestamps)


if __name__ == "__main__":
    main()
