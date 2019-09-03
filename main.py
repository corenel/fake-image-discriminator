import os
import random
import time

import setting
import util

if __name__ == '__main__':
    # initialize
    timestamp = time.strftime("%Y%m%d%H%M%S")
    answer_dict = {}
    real_image_dict = util.get_image_dict(setting.REAL_IMAGE_DIR)
    fake_image_list = util.get_image_list(setting.FAKE_IMAGE_DIR)[:5]
    random.shuffle(fake_image_list)

    # reset status
    current_index = 0
    current_real_samples = util.get_samples(fake_image_list[current_index],
                                            real_image_dict)
    selected_grid_idx = None
    fake_gird_idx = random.choice(range(setting.NUM_DISPLAY))
    keycode = 255

    while keycode != setting.VALID_KEYS['exit']:

        # grid keys
        if keycode in setting.GRID_KEYS:
            selected_grid_idx = keycode - 49

        # operation keys
        if keycode == setting.VALID_KEYS[
            'process'] and selected_grid_idx is not None:
            answer_dict[current_index] = selected_grid_idx == fake_gird_idx
            current_index += 1
            if current_index >= len(fake_image_list):
                break
            current_real_samples = util.get_samples(
                fake_image_list[current_index], real_image_dict)
            selected_grid_idx = None
            fake_gird_idx = random.choice(range(setting.NUM_DISPLAY))

        # read images
        image_list = current_real_samples[:]
        image_list.insert(fake_gird_idx, fake_image_list[current_index])
        images = util.read_images(image_list,
                                  setting.NUM_DISPLAY,
                                  invalid_list=[selected_grid_idx]
                                  if selected_grid_idx is not None else [])
        grid = util.create_image_grid(images,
                                      grid_size=(setting.NUM_DISPLAY, 1))

        # debug
        if setting.DEBUG:
            print(keycode)
            print('curr: {}'.format(current_index))
            print('fake_idx: {}'.format(fake_gird_idx))
            print('selected idx: {}'.format(selected_grid_idx))
            print('Correct: {}'.format(selected_grid_idx == fake_gird_idx))

        # display
        keycode = util.show_image_grid(grid, scale=setting.DSIPALY_SCALE)

    if keycode != setting.VALID_KEYS['exit']:
        os.makedirs(setting.LOG_DIR, exist_ok=True)
        save_path = os.path.join(setting.LOG_DIR,
                                 'answer_{}.log'.format(timestamp))
        with open(save_path, 'w') as f:
            f.write('file,is_fake')
            for fake_image_id, answer in answer_dict.items():
                image_path = fake_image_list[fake_image_id]
                if setting.DEBUG:
                    print(image_path, answer)
                f.write('{},,{}\n'.format(image_path, answer))
        print('Answers saved in {}'.format(save_path))
