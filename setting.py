import os

DEBUG = False

# path
IMAGE_DIR = '/Users/yuthon/Downloads/user'
LOG_DIR = os.path.join(IMAGE_DIR, 'log')
FAKE_IMAGE_DIR = os.path.join(IMAGE_DIR, 'fake')
REAL_IMAGE_DIR = os.path.join(IMAGE_DIR, 'real')
IMAGE_EXT = ['.jpg', '.png']
OTHER_EXT = ['.xml']

# display
NUM_DISPLAY = 3
DSIPALY_SCALE = 1.0

# button
VALID_KEYS = {
    'prev': 104,  # h
    'next': 108,  # l
    'down': 106,  # j
    'up': 107,  # k
    'discard': 120,  # x
    'reserve': 99,  # c
    'process': 13,  # enter
    'exit': 27,  # esc
}
GRID_KEYS = [49 + i for i in range(NUM_DISPLAY)]
