from console import start, move


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
                cells.append(i / 1024 if i else 0)
        return cells

    @property
    def score(self):
        return self.raw_score['score']

    def is_over(self):
        return False

    def move(self, direction):
        move(self.grid, self.raw_score, direction)


# if __name__ == '__main__':
#     for _ in range(3):
#         game.move('up')
