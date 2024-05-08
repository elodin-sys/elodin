
#   3rd party / std lib imports
import pytest
from Basilisk.architecture import messaging
#   Modules to test
from Basilisk.fswAlgorithms import hillStateConverter
#   Utilities/macros
from Basilisk.utilities import SimulationBaseClass as sbc
from Basilisk.utilities import macros


#from Basilisk.simulation import simFswInterfaceMessages

def test_hillStateConverter(show_plots):
    """
    Tests the hillStateConverter module for the following:
    1. Accepts both a hill and deputy message;
    2. Correctly converts those messages into the hill frame.
    """
    sim = sbc.SimBaseClass()
    procName = 'process'
    taskName = 'task'
    proc = sim.CreateNewProcess(procName)
    task =  sim.CreateNewTask(taskName, macros.sec2nano(1.0))
    proc.addTask(task)

    #   Set up two spacecraft position messages
    chief_r = [7100,0,0]
    chief_v = [0,7.000,0]
    dep_r = [7101, 0, 0]
    dep_v = [0,7.010,0]

    chiefNavMsgData = messaging.NavTransMsgPayload()
    chiefNavMsgData.r_BN_N = chief_r
    chiefNavMsgData.v_BN_N = chief_v
    chiefNavMsg = messaging.NavTransMsg().write(chiefNavMsgData)

    depNavMsgData = messaging.NavTransMsgPayload()
    depNavMsgData.r_BN_N = dep_r 
    depNavMsgData.v_BN_N = dep_v
    depNavMsg = messaging.NavTransMsg().write(depNavMsgData)

    #   Set up the hillStateConverter
    hillStateNav = hillStateConverter.hillStateConverter()
    hillStateNav.ModelTag = "dep_hillStateNav"
    hillStateNav.chiefStateInMsg.subscribeTo(chiefNavMsg)
    hillStateNav.depStateInMsg.subscribeTo(depNavMsg)
    hillRecorder = hillStateNav.hillStateOutMsg.recorder()
    
    sim.AddModelToTask(taskName, hillStateNav)
    sim.AddModelToTask(taskName, hillRecorder)

    sim.ConfigureStopTime(macros.sec2nano(1.0))
    sim.InitializeSimulation()
    sim.ExecuteSimulation()

    hill_positions = hillRecorder.r_DC_H
    hill_velocities = hillRecorder.v_DC_H

    ref_pos = [1,0,0]
    ref_vel = [0,0.00901408,0]
    #   Test the position calculation:
    for val1, val2 in zip(hill_positions[-1], ref_pos):
        assert  val1 == pytest.approx(val2)

    for val1, val2  in zip(hill_velocities[-1], ref_vel):
        assert val1 == pytest.approx(val2)

if __name__=="__main__":
    test_hillStateConverter(False)
