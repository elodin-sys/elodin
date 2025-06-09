# elodin-py

Python client for elodin-db, providing a simple interface to send table data to the database using numpy arrays.

## Installation

Build and install the package using maturin:

```bash
uvx maturin develop
```

## Usage

```python
import numpy as np
import elodin

# Create a client
client = elodin.ElodinClient("127.0.0.1:2240")

# Send table data
data = np.array([1.0, 2.0, 3.0, 4.0])
client.send_table(
    entity_id=123,
    component_id="position",
    data=data
)
```

## API

### `ElodinClient(addr: str)`

Creates a new client that will connect to the specified address.

### `send_table(entity_id: int, component_id: str, data: numpy.ndarray)`

Sends a table of data to the database:
- `entity_id`: Unique identifier for the entity
- `component_id`: String identifier for the component type
- `data`: 1D numpy array of float64 values

The method automatically creates and sends the required VTable message to describe the data structure, then sends the actual table data.

## Development

Run tests:

```bash
uv run python -m pytest tests/ -v
```

## Requirements

- Python 3.8+
- numpy >= 1.20.0
