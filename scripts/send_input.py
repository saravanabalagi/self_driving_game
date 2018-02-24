# directx scan codes http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions
def press_key(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_key(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# Keys
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

def send_key_input(key):
    # repeat = 1
    # if key == W or key == S:
        # repeat = 5
    repeat = 5
    for i in range(repeat):
        press_key(key)
    # release_key(key)

def release_all_keys():
    release_key(W)
    release_key(S)
    release_key(A)
    release_key(D)

def label_to_keys(label):
    if   label == 0: release_key(W); release_key(S); release_key(A); release_key(D)
    elif label == 5: release_all_keys(); send_key_input(W); send_key_input(A)
    elif label == 6: release_all_keys(); send_key_input(W); send_key_input(D)
    elif label == 7: release_all_keys(); send_key_input(S); send_key_input(A)
    elif label == 8: release_all_keys(); send_key_input(S); send_key_input(D)
    elif label == 1: release_all_keys(); send_key_input(W)
    elif label == 2: release_all_keys(); send_key_input(S)
    elif label == 3: release_all_keys(); send_key_input(A)
    elif label == 4: release_all_keys(); send_key_input(D)
    else: print("Label Error")