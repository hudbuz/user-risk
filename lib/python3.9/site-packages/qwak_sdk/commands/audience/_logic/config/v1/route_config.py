from dataclasses import dataclass

from _qwak_proto.qwak.audience.v1.audience_pb2 import Route


@dataclass
class RouteConfig:
    variation_name: str
    weight: int
    shadow: bool

    def to_route_api(self):
        return Route(
            variation_name=self.variation_name, weight=self.weight, shadow=self.shadow
        )
