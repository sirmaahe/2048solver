from time import sleep
from selenium import webdriver


class Game:
    def __init__(self):
        self.web_driver = webdriver.Chrome(executable_path='./chromedriver')
        self.web_driver.get("https://gabrielecirulli.github.io/2048/")

    @property
    def elements(self):
        container = self.web_driver.find_element_by_class_name("tile-container")
        container_divs = container.find_elements_by_tag_name('div')

        cells = [0 for _ in range(16)]
        for item in container_divs:
            pos = item.get_attribute("class").split(' ')

            if len(pos) < 3:
                continue

            pos = pos[2]
            pos = int(pos[-3]) - 1 + 4 * (int(pos[-1]) - 1)
            cell = item.text
            if not cell:
                cell = 0
            cells[pos] = int(cell)

        return cells

    @property
    def score(self):
        score = self.web_driver.find_element_by_class_name("score-container").text
        score = score.split('\n')[0]
        return int(score)

    def is_over(self):
        is_over = bool(len(self.web_driver.find_elements_by_css_selector(".game-over")))
        return is_over

    def restart(self):
        self.web_driver.find_element_by_class_name("restart-button").click()

    def move(self, direction):
        code = {
            'up': '\ue013',
            'down': '\ue015',
            'left': '\ue012',
            'right': '\ue014',
        }[direction]
        self.web_driver.find_element_by_tag_name('body').send_keys(code)
        sleep(0.1)

    def close(self):
        self.web_driver.close()


# if __name__ == '__main__':
#     for _ in range(3):
#         game.move('up')
