import pytest
import numpy as np
import elodin_db


def test_client_creation():
    """Test that we can create a client instance"""
    client = elodin_db.ElodinClient("127.0.0.1:2240")
    assert client is not None

    # Invalid address should fail
    with pytest.raises(Exception):
        client = elodin_db.ElodinClient("invalid-address")


def test_send_table_interface():
    test_data = np.array([1.0, 2.0, 3.0, 4.0])
    client = elodin_db.ElodinClient("127.0.0.1:2240")
    client.send_table(123, "test_component", test_data)
