# from ctypes import *
from console import start, move


# lib = cdll.LoadLibrary("./consoleGo.so")
# lib.Start.restype = c_char_p
# lib.Move.argtypes = c_char_p, c_int, c_char_p
# lib.Move.restype = c_char_p


class Game:
    def __init__(self):
        grid, score = start()
        self.grid = grid
        self.raw_score = score

    @property
    def elements(self):
        cells = []
        for item in self.grid:
            for i in item:
                cells.append((i - 1024)/2048 if i else 0)
        return cells

    @property
    def score(self):
        return self.raw_score['score']

    def is_over(self):
        return False

    def move(self, direction):
        move(self.grid, self.raw_score, direction)


# class Game2:
#     def __init__(self):
#         string = lib.Start().decode().split(',')
#         string = [int(v) for v in string]
#         grid = []
#         for i in range(4):
#             grid.append(string[i * 4:(i + 1) * 4])
#         self.grid = grid
#         self.raw_score = 0
#
#     @property
#     def elements(self):
#         cells = []
#         for item in self.grid:
#             for i in item:
#                 cells.append((i - 1024) / 2048 if i else 0)
#         return cells
#
#     @property
#     def score(self):
#         return self.raw_score
#
#     def is_over(self):
#         return False
#
#     def move(self, direction):
#         grid = []
#         for i in self.grid:
#             for j in i:
#                 grid.append(str(j))
#         string = lib.Move((','.join(grid)).encode(), self.raw_score, direction.encode())
#         string = string.decode().split(',')
#         self.raw_score = int(string[-1])
#         del grid[-1]
#         string = [int(v) for v in string]
#         grid = []
#         for i in range(4):
#             grid.append(string[i * 4:(i + 1) * 4])
#         self.grid = grid

# if __name__ == '__main__':
#     for _ in range(3):
#         game.move('up')
