import time
from evdev import UInput, ecodes as e


class VirtualDeviceManager:
    def __init__(self):
        self.virtual_mouse = None
        self.virtual_keyboard = None
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

    def create_devices(self):
        mouse_capabilities = {
            e.EV_REL: [e.REL_X, e.REL_Y, e.REL_WHEEL],
            e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
        }
        self.virtual_mouse = UInput(mouse_capabilities, name='gesture-mouse')

        keyboard_capabilities = {
            e.EV_KEY: [
                e.KEY_LEFTCTRL, e.KEY_LEFTMETA,
                e.KEY_LEFT, e.KEY_RIGHT, e.KEY_UP, e.KEY_DOWN,
                e.KEY_Q,
            ]
        }
        self.virtual_keyboard = UInput(keyboard_capabilities, name='gesture-keyboard')

    def close_devices(self):
        if self.virtual_mouse:
            self.virtual_mouse.close()
            self.virtual_mouse = None
        if self.virtual_keyboard:
            self.virtual_keyboard.close()
            self.virtual_keyboard = None

    def move_mouse(self, target_x, target_y):
        if not self.virtual_mouse:
            return
        dx = int(target_x - self.last_mouse_x)
        dy = int(target_y - self.last_mouse_y)
        if dx != 0:
            self.virtual_mouse.write(e.EV_REL, e.REL_X, dx)
        if dy != 0:
            self.virtual_mouse.write(e.EV_REL, e.REL_Y, dy)
        if dx != 0 or dy != 0:
            self.virtual_mouse.syn()
        self.last_mouse_x, self.last_mouse_y = target_x, target_y

    def press_mouse_button(self, button='left'):
        if not self.virtual_mouse:
            return
        button_code = e.BTN_LEFT if button == 'left' else e.BTN_RIGHT
        self.virtual_mouse.write(e.EV_KEY, button_code, 1)
        self.virtual_mouse.syn()

    def release_mouse_button(self, button='left'):
        if not self.virtual_mouse:
            return
        button_code = e.BTN_LEFT if button == 'left' else e.BTN_RIGHT
        self.virtual_mouse.write(e.EV_KEY, button_code, 0)
        self.virtual_mouse.syn()

    def scroll(self, amount):
        if not self.virtual_mouse or amount == 0:
            return
        self.virtual_mouse.write(e.EV_REL, e.REL_WHEEL, amount)
        self.virtual_mouse.syn()

    def send_workspace_switch(self, direction):
        if not self.virtual_keyboard:
            return
        key_map = {
            'left': e.KEY_LEFT,
            'right': e.KEY_RIGHT,
            'up': e.KEY_UP,
            'down': e.KEY_DOWN,
        }
        if direction not in key_map:
            return

        arrow_key = key_map[direction]
        self.virtual_keyboard.write(e.EV_KEY, e.KEY_LEFTCTRL, 1)
        self.virtual_keyboard.write(e.EV_KEY, e.KEY_LEFTMETA, 1)
        self.virtual_keyboard.write(e.EV_KEY, arrow_key, 1)
        self.virtual_keyboard.syn()
        time.sleep(0.05)
        self.virtual_keyboard.write(e.EV_KEY, arrow_key, 0)
        self.virtual_keyboard.write(e.EV_KEY, e.KEY_LEFTMETA, 0)
        self.virtual_keyboard.write(e.EV_KEY, e.KEY_LEFTCTRL, 0)
        self.virtual_keyboard.syn()

    def send_close_window(self):
        if not self.virtual_keyboard:
            return
        self.virtual_keyboard.write(e.EV_KEY, e.KEY_LEFTMETA, 1)
        self.virtual_keyboard.write(e.EV_KEY, e.KEY_Q, 1)
        self.virtual_keyboard.syn()
        time.sleep(0.05)
        self.virtual_keyboard.write(e.EV_KEY, e.KEY_Q, 0)
        self.virtual_keyboard.write(e.EV_KEY, e.KEY_LEFTMETA, 0)
        self.virtual_keyboard.syn()
