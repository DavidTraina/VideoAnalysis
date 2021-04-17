from dataclasses import dataclass


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
