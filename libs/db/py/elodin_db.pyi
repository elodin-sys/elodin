from typing import Any

import numpy as np
import numpy.typing as npt

class ElodinClient:
    """Client for connecting to and sending data to an Elodin database server."""

    def __init__(self, addr: str) -> None:
        """
        Create a new ElodinClient connection.

        Args:
            addr: Socket address in format "host:port" (e.g., "127.0.0.1:8080")

        Raises:
            RuntimeError: If the address is invalid or connection fails
        """
        ...

    def send_table(
        self,
        entity_id: int,
        component_id: str,
        data: npt.NDArray[Any]
    ) -> None:
        """
        Send a table of data to the Elodin database.

        Args:
            entity_id: The entity ID (u64)
            component_id: The component identifier string
            data: A numpy array of any shape and supported dtype

        Raises:
            RuntimeError: If sending the vtable or table data fails
            RuntimeError: If the array dtype is not supported

        Note:
            Supported dtypes are: f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, bool
            The array must be C-contiguous.
        """
        ...
