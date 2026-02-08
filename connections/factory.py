"""
Connection factory for creating connection instances.

Provides factory methods for creating connections based on type
and for auto-generating connections based on node relationships.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from .base import Connection, ConnectionType
from .electricity import ElectricityConnection
from .hydrogen import HydrogenConnection

# Import node types - try relative first, then absolute
try:
    from ..nodes.base import Carrier, NodeType
    from ..nodes.consumer import ConsumerType
except ImportError:
    from nodes.base import Carrier, NodeType
    from nodes.consumer import ConsumerType

if TYPE_CHECKING:
    from ..nodes import BaseNode, Generator, Transformer, Consumer, Battery, H2Storage


class ConnectionFactory:
    """
    Factory for creating and managing network connections.

    Provides methods to:
    - Create connections from dictionaries
    - Auto-generate connections based on node types
    - Validate connection compatibility
    """

    @staticmethod
    def create(data: Dict[str, Any]) -> Connection:
        """
        Create appropriate connection type from dictionary.

        Args:
            data: Dictionary with connection attributes

        Returns:
            Connection instance of the appropriate type
        """
        carrier = data.get('carrier', 'electricity')
        if isinstance(carrier, ConnectionType):
            carrier = carrier.value

        connection_type = data.get('connection_type', carrier)

        if connection_type == 'hydrogen' or carrier == 'hydrogen':
            return HydrogenConnection.from_dict(data)
        else:
            return ElectricityConnection.from_dict(data)

    @staticmethod
    def create_electricity(source: str, target: str,
                          capacity: Optional[float] = None,
                          enabled: bool = True,
                          loss: float = 0.0) -> ElectricityConnection:
        """Create an electricity connection."""
        return ElectricityConnection(
            _source=source,
            _target=target,
            _capacity=capacity,
            _enabled=enabled,
            _loss=loss,
        )

    @staticmethod
    def create_hydrogen(source: str, target: str,
                       capacity: Optional[float] = None,
                       enabled: bool = True,
                       loss: float = 0.0) -> HydrogenConnection:
        """Create a hydrogen connection."""
        return HydrogenConnection(
            _source=source,
            _target=target,
            _capacity=capacity,
            _enabled=enabled,
            _loss=loss,
        )

    @classmethod
    def auto_generate(cls,
                      generators: List['Generator'],
                      transformers: List['Transformer'],
                      consumers: List['Consumer'],
                      batteries: Optional[List['Battery']] = None,
                      h2_storage: Optional[List['H2Storage']] = None) -> List[Connection]:
        """
        Auto-generate connections based on node types and carriers.

        Connection rules:
        1. Generators → Transformers (electricity)
        2. Generators → Electricity Consumers (electricity)
        3. Generators → Batteries (electricity)
        4. Transformers → Hydrogen Consumers (hydrogen)
        5. Transformers → H2 Storage (hydrogen)
        6. Batteries → Transformers (electricity)
        7. Batteries → Electricity Consumers (electricity)
        8. H2 Storage → Hydrogen Consumers (hydrogen)

        Args:
            generators: List of generator nodes
            transformers: List of transformer nodes
            consumers: List of consumer nodes
            batteries: List of battery nodes (optional)
            h2_storage: List of H2 storage nodes (optional)

        Returns:
            List of auto-generated connections
        """
        connections = []
        batteries = batteries or []
        h2_storage = h2_storage or []

        # Identify consumer types
        electricity_consumers = [c for c in consumers
                                if c.carrier == Carrier.ELECTRICITY or
                                c.consumer_type == ConsumerType.LOGARITHMIC_ELECTRICITY]
        hydrogen_consumers = [c for c in consumers
                             if c.carrier == Carrier.HYDROGEN or
                             c.consumer_type == ConsumerType.LOGARITHMIC_HYDROGEN]
        dual_consumers = [c for c in consumers
                         if c.carrier == Carrier.BOTH or
                         c.consumer_type == ConsumerType.COBB_DOUGLAS]

        # 1. Generators → Transformers (electricity)
        for gen in generators:
            for trans in transformers:
                connections.append(cls.create_electricity(
                    source=gen.name,
                    target=trans.name,
                ))

        # 2. Generators → Electricity Consumers (electricity)
        for gen in generators:
            for consumer in electricity_consumers:
                connections.append(cls.create_electricity(
                    source=gen.name,
                    target=consumer.name,
                ))

        # 3. Generators → Dual Consumers (electricity part)
        for gen in generators:
            for consumer in dual_consumers:
                connections.append(cls.create_electricity(
                    source=gen.name,
                    target=consumer.name,
                ))

        # 4. Generators → Batteries (electricity)
        for gen in generators:
            for battery in batteries:
                connections.append(cls.create_electricity(
                    source=gen.name,
                    target=battery.name,
                ))

        # 5. Transformers → Hydrogen Consumers (hydrogen)
        for trans in transformers:
            for consumer in hydrogen_consumers:
                connections.append(cls.create_hydrogen(
                    source=trans.name,
                    target=consumer.name,
                ))

        # 6. Transformers → Dual Consumers (hydrogen part)
        for trans in transformers:
            for consumer in dual_consumers:
                connections.append(cls.create_hydrogen(
                    source=trans.name,
                    target=consumer.name,
                ))

        # 7. Transformers → H2 Storage (hydrogen)
        for trans in transformers:
            for storage in h2_storage:
                connections.append(cls.create_hydrogen(
                    source=trans.name,
                    target=storage.name,
                ))

        # 8. Batteries → Transformers (electricity)
        for battery in batteries:
            for trans in transformers:
                connections.append(cls.create_electricity(
                    source=battery.name,
                    target=trans.name,
                ))

        # 9. Batteries → Electricity Consumers (electricity)
        for battery in batteries:
            for consumer in electricity_consumers:
                connections.append(cls.create_electricity(
                    source=battery.name,
                    target=consumer.name,
                ))

        # 10. Batteries → Dual Consumers (electricity part)
        for battery in batteries:
            for consumer in dual_consumers:
                connections.append(cls.create_electricity(
                    source=battery.name,
                    target=consumer.name,
                ))

        # 11. H2 Storage → Hydrogen Consumers (hydrogen)
        for storage in h2_storage:
            for consumer in hydrogen_consumers:
                connections.append(cls.create_hydrogen(
                    source=storage.name,
                    target=consumer.name,
                ))

        # 12. H2 Storage → Dual Consumers (hydrogen part)
        for storage in h2_storage:
            for consumer in dual_consumers:
                connections.append(cls.create_hydrogen(
                    source=storage.name,
                    target=consumer.name,
                ))

        return connections

    @staticmethod
    def validate_connection(connection: Connection,
                           source_node: 'BaseNode',
                           target_node: 'BaseNode') -> bool:
        """
        Validate that a connection is compatible with its nodes.

        Args:
            connection: The connection to validate
            source_node: The source node
            target_node: The target node

        Returns:
            True if valid

        Raises:
            ValueError: If connection is invalid
        """
        # Check names match
        if connection.source != source_node.name:
            raise ValueError(f"Connection source '{connection.source}' "
                           f"doesn't match node '{source_node.name}'")
        if connection.target != target_node.name:
            raise ValueError(f"Connection target '{connection.target}' "
                           f"doesn't match node '{target_node.name}'")

        # Check carrier compatibility
        carrier = Carrier.ELECTRICITY if connection.carrier == ConnectionType.ELECTRICITY else Carrier.HYDROGEN

        # Generators can only output electricity
        if source_node.node_type == NodeType.GENERATOR and carrier != Carrier.ELECTRICITY:
            raise ValueError("Generators can only output electricity")

        # Batteries can only handle electricity
        if source_node.node_type == NodeType.BATTERY and carrier != Carrier.ELECTRICITY:
            raise ValueError("Batteries can only handle electricity")
        if target_node.node_type == NodeType.BATTERY and carrier != Carrier.ELECTRICITY:
            raise ValueError("Batteries can only handle electricity")

        # H2 Storage can only handle hydrogen
        if source_node.node_type == NodeType.H2_STORAGE and carrier != Carrier.HYDROGEN:
            raise ValueError("H2 Storage can only handle hydrogen")
        if target_node.node_type == NodeType.H2_STORAGE and carrier != Carrier.HYDROGEN:
            raise ValueError("H2 Storage can only handle hydrogen")

        return True

    @staticmethod
    def serialize_connections(connections: List[Connection]) -> List[Dict[str, Any]]:
        """Serialize a list of connections to dictionaries."""
        return [conn.to_dict() for conn in connections]

    @classmethod
    def deserialize_connections(cls, data: List[Dict[str, Any]]) -> List[Connection]:
        """Deserialize a list of dictionaries to connections."""
        return [cls.create(d) for d in data]
