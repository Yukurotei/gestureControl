import ctypes
import ctypes.wintypes
import time

user32 = ctypes.windll.user32

# Input type constants
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

# Mouse event flags
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_MOVE_NOCOALESCE = 0x2000

# Keyboard event flags
KEYEVENTF_KEYUP = 0x0002

# Virtual key codes
VK_CONTROL = 0xA2
VK_LWIN = 0x5B
VK_MENU = 0xA4  # Alt
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_F4 = 0x73

WHEEL_DELTA = 120


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("union", _INPUT_UNION),
    ]


def _send_input(*inputs):
    n = len(inputs)
    arr = (INPUT * n)(*inputs)
    user32.SendInput(n, arr, ctypes.sizeof(INPUT))


def _mouse_input(dx=0, dy=0, mouse_data=0, flags=0):
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi.dx = dx
    inp.union.mi.dy = dy
    inp.union.mi.mouseData = mouse_data
    inp.union.mi.dwFlags = flags
    return inp


def _key_input(vk, up=False):
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki.wVk = vk
    inp.union.ki.dwFlags = KEYEVENTF_KEYUP if up else 0
    return inp


class VirtualDeviceManager:
    def __init__(self):
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

    def create_devices(self):
        pass  # No virtual devices needed on Windows

    def close_devices(self):
        pass  # Nothing to clean up

    def move_mouse(self, target_x, target_y):
        dx = int(target_x - self.last_mouse_x)
        dy = int(target_y - self.last_mouse_y)
        if dx != 0 or dy != 0:
            _send_input(_mouse_input(
                dx=dx, dy=dy,
                flags=MOUSEEVENTF_MOVE | MOUSEEVENTF_MOVE_NOCOALESCE,
            ))
        self.last_mouse_x, self.last_mouse_y = target_x, target_y

    def press_mouse_button(self, button='left'):
        flag = MOUSEEVENTF_LEFTDOWN if button == 'left' else MOUSEEVENTF_RIGHTDOWN
        _send_input(_mouse_input(flags=flag))

    def release_mouse_button(self, button='left'):
        flag = MOUSEEVENTF_LEFTUP if button == 'left' else MOUSEEVENTF_RIGHTUP
        _send_input(_mouse_input(flags=flag))

    def scroll(self, amount):
        if amount == 0:
            return
        _send_input(_mouse_input(
            mouse_data=amount * WHEEL_DELTA,
            flags=MOUSEEVENTF_WHEEL,
        ))

    def send_workspace_switch(self, direction):
        key_map = {
            'left': VK_LEFT,
            'right': VK_RIGHT,
            'up': VK_UP,
            'down': VK_DOWN,
        }
        if direction not in key_map:
            return

        vk_arrow = key_map[direction]
        # Ctrl+Win+Arrow for virtual desktop switching
        _send_input(
            _key_input(VK_CONTROL),
            _key_input(VK_LWIN),
            _key_input(vk_arrow),
        )
        time.sleep(0.05)
        _send_input(
            _key_input(vk_arrow, up=True),
            _key_input(VK_LWIN, up=True),
            _key_input(VK_CONTROL, up=True),
        )

    def send_close_window(self):
        # Alt+F4 to close the foreground window
        _send_input(
            _key_input(VK_MENU),
            _key_input(VK_F4),
        )
        time.sleep(0.05)
        _send_input(
            _key_input(VK_F4, up=True),
            _key_input(VK_MENU, up=True),
        )
