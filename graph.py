from dataclass import dataclass

@dataclass
class ObjectNode:
    id: int
    label: str
    bbox: tuple[float]
    conf: float
    attrs: dict[str, str]

class Graph:
    def __init__(self, image_url: str, size: list[int]) -> None:
        self.image_url = image_url
        self.size = size
        self._objects: dict[int, ObjectNode] = []
    
    def add_object(self, obj: ObjectNode) -> None:
        self._objects[obj.id] = obj  
    
    def remove_object(self, obj_id: int) -> None:
        if obj_id in self._objects:
            del self._objects[obj_id]
        else:
            raise ValueError(f"Object with id {obj_id} not found.")
    
    def get_objects(self) -> list[ObjectNode]:
        return [obj for obj in self._objects.values()]

    def get_object_by_id(self, obj_id: int) -> ObjectNode:
        if obj_id in self._objects:
            return self._objects[obj_id]
        else:
            raise ValueError(f"Object with id {obj_id} not found.")
    