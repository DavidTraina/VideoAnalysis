from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from functools import cached_property


class Gender(Enum):
    FEMALE = 0
    MALE = 1
    UNKNOWN = 3


Point = namedtuple('Point', ['x', 'y'])


@dataclass
class Box:
    top_left: Point
    bottom_right: Point

    @classmethod
    def from_points(cls, top_left, bottom_right):
        return cls(top_left=top_left, bottom_right=bottom_right)

    @classmethod
    def from_values(cls, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
        return cls(
            top_left=Point(x=top_left_x, y=top_left_y),
            bottom_right=Point(x=bottom_right_x, y=bottom_right_y),
        )

    @property
    def top_left_x(self):
        return self.top_left.x

    @property
    def top_left_y(self):
        return self.top_left.y

    @property
    def bottom_right_x(self) -> int:
        return self.bottom_right.x

    @property
    def bottom_right_y(self) -> int:
        return self.bottom_right.y

    @cached_property
    def center(self) -> Point:
        return Point(
            x=(self.top_left.x + self.bottom_right.x) / 2,
            y=(self.top_left.y + self.bottom_right.y) / 2,
        )


@dataclass(frozen=True)
class Color:
    r: int = 0
    b: int = 0
    g: int = 0

    def to_bgr(self):
        return self.b, self.g, self.r

    def to_rgb(self):
        return self.r, self.g, self.b

    @classmethod
    def black(cls) -> 'Color':
        return cls()

    @classmethod
    def white(cls) -> 'Color':
        return cls(255, 255, 255)

    @classmethod
    def red(cls) -> 'Color':
        return cls(r=255)

    @classmethod
    def green(cls) -> 'Color':
        return cls(g=255)

    @classmethod
    def blue(cls) -> 'Color':
        return cls(b=255)

    @classmethod
    def yellow(cls) -> 'Color':
        return cls(r=255, g=255)
