from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt

class ElodinDB:
    """Client for connecting to and sending data to an Elodin database server."""

    @staticmethod
    def start(addr: str = "[::]:0", path = "/tmp/db") -> ElodinDB:
        """
        Starts a new instance of elodin-db and returns a client connected to it
        """
        ...


    @staticmethod
    def connect(addr: str) -> ElodinDB:
        """
        Create a new ElodinDB connection.

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

    def set_component_metadata(
        self,
        component_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        asset: Optional[bool] = None
    ) -> None:
        """
        Set metadata for a component.

        Args:
            component_id: The component identifier
            name: Human-readable name for the component (defaults to component_id if not provided)
            metadata: Optional dictionary of metadata key-value pairs
            asset: Optional flag indicating if this component is an asset

        Raises:
            RuntimeError: If sending the metadata fails

        Example:
            db.set_component_metadata("position", "Position",
                                      metadata={"units": "meters", "frame": "ECEF"},
                                      asset=False)
            # or with name defaulting to component_id:
            db.set_component_metadata("position",
                                      metadata={"element_names": "x,y,z"})
        """
        ...

    def set_entity_metadata(
        self,
        entity_id: int,
        name: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Set metadata for an entity.

        Args:
            entity_id: The numeric entity identifier
            name: Human-readable name for the entity
            metadata: Optional dictionary of metadata key-value pairs

        Raises:
            RuntimeError: If sending the metadata fails

        Example:
            db.set_entity_metadata(42, "Satellite Alpha",
                                   metadata={"type": "LEO", "mission": "Earth observation"})
        """
        ...
