from runner import *
from tensorflow import keras
import numpy as np
model = keras.models.load_model("./model")

cellWidth = 10
cellHeight = 10
rows = 28
cols = 28
grid = [[False] * cols for _ in range(rows)] # grid[r][c]
color = True

def mouse_ind():
    mx, my = get_mouse_pos()
    return my // cellHeight, mx // cellWidth

def get_ans(arr):
    return np.argmax(model(arr.reshape(1, rows, cols), training=False).numpy())

class Main(Application):
    def draw(self):
        global grid
        fill(255, 255, 255)
        rect(0, 0, cols * cellWidth, rows * cellHeight)

        if mouse_down(3)[0]:
            r, c = mouse_ind()
            grid[r][c] = color
        if mouse_down(3)[2]:
            grid = [[False] * cols for _ in range(rows)]

        fill(0, 0, 0)
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]:
                    rect(cellWidth * c, cellHeight * r, cellWidth, cellHeight)

        print(get_ans(np.array(grid).astype(float)))

    def mouse_pressed(self, event):
        global color
        r, c = mouse_ind()
        color = not grid[r][c]


Main(cols * cellWidth, rows * cellHeight)