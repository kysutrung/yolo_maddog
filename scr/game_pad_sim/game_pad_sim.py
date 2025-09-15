# win_virtual_gamepad_tk.py
import tkinter as tk
from tkinter import ttk
import vgamepad as vg

# ------- Helpers -------
def lerp(val, src_min, src_max, dst_min, dst_max):
    return int(dst_min + (val - src_min) * (dst_max - dst_min) / (src_max - src_min))

def scale_axis(v_0_100):
    # Tkinter slider 0..100 -> XInput -32768..32767
    return lerp(v_0_100, 0, 100, -32768, 32767)

def scale_trigger(v_0_100):
    # 0..100 -> 0..255
    return lerp(v_0_100, 0, 100, 0, 255)

# ------- App -------
class GamepadUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Virtual Xbox 360 Gamepad (ViGEm + vgamepad + Tkinter)")
        self.geometry("620x420")

        # Tạo gamepad ảo
        self.pad = vg.VX360Gamepad()

        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # ---- Left stick (X,Y) ----
        lf_axes = ttk.LabelFrame(root, text="Left Stick")
        lf_axes.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self.sld_x = ttk.Scale(lf_axes, from_=0, to=100, orient="horizontal")
        self.sld_x.set(50)
        self.sld_y = ttk.Scale(lf_axes, from_=0, to=100, orient="horizontal")
        self.sld_y.set(50)

        ttk.Label(lf_axes, text="X").grid(row=0, column=0, sticky="w")
        self.sld_x.grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        ttk.Label(lf_axes, text="Y").grid(row=1, column=0, sticky="w")
        self.sld_y.grid(row=1, column=1, sticky="ew", padx=6, pady=3)

        lf_axes.columnconfigure(1, weight=1)

        # ---- Triggers ----
        lf_tr = ttk.LabelFrame(root, text="Triggers")
        lf_tr.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        self.sld_lt = ttk.Scale(lf_tr, from_=0, to=100, orient="horizontal")
        self.sld_rt = ttk.Scale(lf_tr, from_=0, to=100, orient="horizontal")
        ttk.Label(lf_tr, text="LT").grid(row=0, column=0, sticky="w")
        self.sld_lt.grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        ttk.Label(lf_tr, text="RT").grid(row=1, column=0, sticky="w")
        self.sld_rt.grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        lf_tr.columnconfigure(1, weight=1)

        # ---- Buttons ----
        lf_btn = ttk.LabelFrame(root, text="Buttons")
        lf_btn.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        self.varA = tk.BooleanVar()
        self.varB = tk.BooleanVar()
        self.varX = tk.BooleanVar()
        self.varY = tk.BooleanVar()
        self.varLB = tk.BooleanVar()
        self.varRB = tk.BooleanVar()
        self.varStart = tk.BooleanVar()
        self.varBack = tk.BooleanVar()

        ttk.Checkbutton(lf_btn, text="A", variable=self.varA).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(lf_btn, text="B", variable=self.varB).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(lf_btn, text="X", variable=self.varX).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(lf_btn, text="Y", variable=self.varY).grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(lf_btn, text="LB", variable=self.varLB).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(lf_btn, text="RB", variable=self.varRB).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(lf_btn, text="Start", variable=self.varStart).grid(row=1, column=2, sticky="w")
        ttk.Checkbutton(lf_btn, text="Back", variable=self.varBack).grid(row=1, column=3, sticky="w")

        for c in range(4):
            lf_btn.columnconfigure(c, weight=1)

        # ---- D-pad ----
        lf_dpad = ttk.LabelFrame(root, text="D-Pad")
        lf_dpad.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)

        self.varUp = tk.BooleanVar()
        self.varDown = tk.BooleanVar()
        self.varLeft = tk.BooleanVar()
        self.varRight = tk.BooleanVar()
        ttk.Checkbutton(lf_dpad, text="Up", variable=self.varUp).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(lf_dpad, text="Left", variable=self.varLeft).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(lf_dpad, text="Right", variable=self.varRight).grid(row=1, column=2, sticky="w")
        ttk.Checkbutton(lf_dpad, text="Down", variable=self.varDown).grid(row=2, column=1, sticky="w")
        for c in range(3):
            lf_dpad.columnconfigure(c, weight=1)

        # ---- Polling loop (≈60 Hz) ----
        self.after(16, self._tick)

        # Đặt tỉ lệ co giãn
        for i in range(2):
            root.rowconfigure(i, weight=1)
        for j in range(2):
            root.columnconfigure(j, weight=1)

        # Đặt lại về mặc định khi đóng
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _tick(self):
        # Axes
        x = scale_axis(self.sld_x.get())
        y = scale_axis(self.sld_y.get())
        self.pad.left_joystick(x_value=x, y_value=y)

        # Triggers
        lt = scale_trigger(self.sld_lt.get())
        rt = scale_trigger(self.sld_rt.get())
        self.pad.left_trigger(value=lt)
        self.pad.right_trigger(value=rt)

        # Buttons map
        self._set_button(self.varA, vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self._set_button(self.varB, vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        self._set_button(self.varX, vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        self._set_button(self.varY, vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        self._set_button(self.varLB, vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        self._set_button(self.varRB, vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        self._set_button(self.varStart, vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
        self._set_button(self.varBack, vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)

        # D-pad (dạng button riêng lẻ trong XInput)
        self._set_button(self.varUp, vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        self._set_button(self.varDown, vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
        self._set_button(self.varLeft, vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        self._set_button(self.varRight, vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)

        # Gửi state tới driver
        self.pad.update()

        # Lặp tiếp
        self.after(16, self._tick)

    def _set_button(self, var, button_const):
        if var.get():
            self.pad.press_button(button=button_const)
        else:
            self.pad.release_button(button=button_const)

    def on_close(self):
        try:
            # Đưa mọi thứ về neutral
            self.pad.reset()
            self.pad.update()
        except Exception:
            pass
        self.destroy()

if __name__ == "__main__":
    GamepadUI().mainloop()
