# gamepad.py
from dataclasses import dataclass
from enum import Enum

from .config import AX_LX, AX_LY, AX_RX, AX_RY, DEADZONE, SWAP_REAL_LR

try:
    import pygame
except ImportError:
    pygame = None

try:
    import vgamepad as vg
except ImportError:
    vg = None

def dz(v, d=DEADZONE):
    return 0.0 if abs(v) < d else v

class ForwardState(str, Enum):
    OFF = "Auto"
    ARMING = "Arming"
    ON = "Manual"

class GamepadBridge:
    def __init__(self):
        self.pad_name = "N/A"
        self.pygame_ok = False
        self.vpad_ok = False

        self.state = ForwardState.ON   # DEFAULT = Manual
        self.v_lx = self.v_ly = self.v_rx = self.v_ry = 0.0
        self.r_lx = self.r_ly = self.r_rx = self.r_ry = 0.0

        # init pygame joystick
        if pygame:
            try:
                pygame.init()
                pygame.joystick.init()
                if pygame.joystick.get_count() > 0:
                    self.js = pygame.joystick.Joystick(0)
                    self.js.init()
                    self.pad_name = self.js.get_name()
                    self.pygame_ok = True
            except Exception as e:
                print(e)

        # init virtual gamepad
        if vg:
            try:
                self.vpad = vg.VX360Gamepad()
                self.vpad_ok = True
            except Exception as e:
                print(e)

    def read_axes_real(self):
        if not self.pygame_ok:
            return (0, 0, 0, 0)
        try:
            pygame.event.pump()
            if SWAP_REAL_LR:
                lx, ly, rx, ry = [dz(self.js.get_axis(a)) for a in [AX_RX, AX_RY, AX_LX, AX_LY]]
            else:
                lx, ly, rx, ry = [dz(self.js.get_axis(a)) for a in [AX_LX, AX_LY, AX_RX, AX_RY]]
        except Exception:
            lx = ly = rx = ry = 0.0

        self.r_lx, self.r_ly, self.r_rx, self.r_ry = [max(-1, min(1, float(v))) for v in [lx, ly, rx, ry]]
        return self.r_lx, self.r_ly, self.r_rx, self.r_ry

    def send_to_virtual(self, lx, ly, rx, ry):
        self.v_lx, self.v_ly, self.v_rx, self.v_ry = lx, ly, rx, ry
        if not self.vpad_ok:
            return
        self.vpad.left_joystick_float(lx, ly)
        self.vpad.right_joystick_float(rx, ry)
        self.vpad.update()
