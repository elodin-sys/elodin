import pytest
import numpy as np
import elodin_db


def test_client_creation():
    """Test that we can create a client instance"""
    client = elodin_db.ElodinDB.start()
    assert client is not None

    # Invalid address should fail
    with pytest.raises(Exception):
        client = elodin_db.ElodinDB("invalid-address")


def test_send_table_interface():
    test_data = np.array([1.0, 2.0, 3.0, 4.0])
    client = elodin_db.ElodinDB.start()
    client.send_table(123, "test_component", test_data)

def test_metadata():
    """Test that metadata can be set alongside actual data."""
    # Start a test database
    db = elodin_db.ElodinDB.start()

    # Set component metadata
    db.set_component_metadata(
        "position",
        "Position",
        metadata={"units": "meters", "frame": "ECEF"}
    )

    # Set entity metadata
    db.set_entity_metadata(1, "Satellite-1", metadata={"type": "LEO"})

    # Send some data for the component
    position_data = np.array([1000.0, 2000.0, 3000.0], dtype=np.float64)
    db.send_table(1, "position", position_data)

    # Test with multiple components and entities
    db.set_component_metadata("velocity", "Velocity", metadata={"element_names": "x,y,z"})
    db.set_entity_metadata(2, "Satellite-2", metadata={"type": "GEO"})
