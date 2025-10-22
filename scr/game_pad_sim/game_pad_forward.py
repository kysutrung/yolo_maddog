import sys
import tkinter as tk
from tkinter import messagebox

# --- vgamepad / ViGEm ---
try:
    import vgamepad as vg
except ImportError:
    print("Chưa cài 'vgamepad'. Hãy chạy: pip install vgamepad")
    sys.exit(1)

# --- pygame chỉ để đọc tay cầm thật ---
import pygame

# -------- Mapping gốc trên thiết bị --------
AX_LX = 0  # A00
AX_LY = 1  # A01
AX_RY = 2  # A02
AX_RX = 3  # A03

DEADZONE   = 0.08   # 0.0 nếu muốn thấy cả nhiễu nhỏ
POLL_MS    = 8      # chu kỳ đọc (ms) ~125Hz
ARM_EPS    = 0.05   # sai số cho ARMING/Sync

# ✅ Sửa tay cầm thật đang bị ngược trái/phải:
SWAP_REAL_LR = True

def dz(v, d=DEADZONE):
    return 0.0 if abs(v) < d else v

class ForwardState:
    OFF = "OFF"
    ARMING = "ARMING"
    ON = "ON"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Forward Joysticks → vgamepad (Tkinter)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.resizable(False, False)  # 🔒 Cố định kích thước cửa sổ

        # --- Khởi tạo pygame (joystick only, không mở cửa sổ) ---
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            messagebox.showerror("Lỗi", "Không tìm thấy tay cầm thật. Hãy cắm tay cầm rồi chạy lại.")
            self.root.destroy()
            sys.exit(2)
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        self.pad_name = self.js.get_name()

        # --- Khởi tạo tay cầm ảo ---
        try:
            self.vpad = vg.VX360Gamepad()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không khởi tạo được tay cầm ảo.\n{e}")
            self.root.destroy()
            sys.exit(3)

        # --- Trạng thái ---
        self.state = ForwardState.OFF
        self.invert_y_var = tk.BooleanVar(value=False)

        # Trạng thái tay ảo hiện tại (để hiển thị Virtual)
        self.v_lx = 0.0
        self.v_ly = 0.0
        self.v_rx = 0.0
        self.v_ry = 0.0

        # --- UI ---
        wrap = tk.Frame(root, padx=12, pady=12)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text=f"Tay cầm thật: {self.pad_name}", font=("Consolas", 11)).pack(anchor="w")

        opts = tk.Frame(wrap)
        opts.pack(anchor="w", pady=(6, 2))
        self.status_lbl = tk.Label(opts, text="Forward: OFF", font=("Consolas", 12, "bold"), fg="#c33")
        self.status_lbl.pack(side="left")
        tk.Checkbutton(opts, text="Invert Y", variable=self.invert_y_var).pack(side="left", padx=(16,0))

        # Real
        frm_real = tk.LabelFrame(wrap, text="Real Controller (tay cầm thật)", padx=8, pady=6)
        frm_real.pack(fill="x", pady=(10, 6))
        self.lbl_lx_real = tk.Label(frm_real, text="LX: +0.000", font=("Consolas", 14))
        self.lbl_ly_real = tk.Label(frm_real, text="LY: +0.000", font=("Consolas", 14))
        self.lbl_rx_real = tk.Label(frm_real, text="RX: +0.000", font=("Consolas", 14))
        self.lbl_ry_real = tk.Label(frm_real, text="RY: +0.000", font=("Consolas", 14))
        self.lbl_lx_real.grid(row=0, column=0, padx=(0,20), pady=2, sticky="w")
        self.lbl_ly_real.grid(row=1, column=0, padx=(0,20), pady=2, sticky="w")
        self.lbl_rx_real.grid(row=0, column=1, padx=(0,20), pady=2, sticky="w")
        self.lbl_ry_real.grid(row=1, column=1, padx=(0,20), pady=2, sticky="w")

        # Virtual
        frm_virt = tk.LabelFrame(wrap, text="Virtual Controller (tay cầm ảo)", padx=8, pady=6)
        frm_virt.pack(fill="x", pady=(0, 6))
        self.lbl_lx_virt = tk.Label(frm_virt, text="LX: +0.000", font=("Consolas", 14))
        self.lbl_ly_virt = tk.Label(frm_virt, text="LY: +0.000", font=("Consolas", 14))
        self.lbl_rx_virt = tk.Label(frm_virt, text="RX: +0.000", font=("Consolas", 14))
        self.lbl_ry_virt = tk.Label(frm_virt, text="RY: +0.000", font=("Consolas", 14))
        self.lbl_lx_virt.grid(row=0, column=0, padx=(0,20), pady=2, sticky="w")
        self.lbl_ly_virt.grid(row=1, column=0, padx=(0,20), pady=2, sticky="w")
        self.lbl_rx_virt.grid(row=0, column=1, padx=(0,20), pady=2, sticky="w")
        self.lbl_ry_virt.grid(row=1, column=1, padx=(0,20), pady=2, sticky="w")

        # Buttons
        btns = tk.Frame(wrap)
        btns.pack(fill="x", pady=(6, 0))
        self.toggle_btn = tk.Button(btns, text="Forward ON", command=self.toggle_forward)
        self.toggle_btn.pack(side="left")
        tk.Button(btns, text="Quit", command=self.on_close).pack(side="right")

        # start loop
        self.tick()

    # ---- State machine ----
    def toggle_forward(self):
        if self.state == ForwardState.OFF:
            self.state = ForwardState.ARMING
            self.status_lbl.config(text="Forward: ARMING", fg="#cc0")
            self.toggle_btn.config(text="Cancel")
        elif self.state == ForwardState.ARMING:
            self.state = ForwardState.OFF
            self.status_lbl.config(text="Forward: OFF", fg="#c33")
            self.toggle_btn.config(text="Forward ON")
        elif self.state == ForwardState.ON:
            self.state = ForwardState.OFF
            self.status_lbl.config(text="Forward: OFF", fg="#c33")
            self.toggle_btn.config(text="Forward ON")

    # ---- I/O ----
    def read_axes_real(self):
        """Đọc tay cầm thật và sửa L/R nếu cần"""
        pygame.event.pump()
        try:
            if SWAP_REAL_LR:
                lx = dz(self.js.get_axis(AX_RX))
                ly = dz(self.js.get_axis(AX_RY))
                rx = dz(self.js.get_axis(AX_LX))
                ry = dz(self.js.get_axis(AX_LY))
            else:
                lx = dz(self.js.get_axis(AX_LX))
                ly = dz(self.js.get_axis(AX_LY))
                rx = dz(self.js.get_axis(AX_RX))
                ry = dz(self.js.get_axis(AX_RY))
        except Exception:
            lx = ly = rx = ry = 0.0

        if self.invert_y_var.get():
            ly = -ly
            ry = -ry

        lx = max(-1.0, min(1.0, lx))
        ly = max(-1.0, min(1.0, ly))
        rx = max(-1.0, min(1.0, rx))
        ry = max(-1.0, min(1.0, ry))
        return lx, ly, rx, ry

    def send_to_virtual(self, lx, ly, rx, ry):
        """Left ảo ⇐ (LX, LY), Right ảo ⇐ (RX, RY)"""
        try:
            self.vpad.left_joystick_float(x_value_float=lx, y_value_float=ly)
            self.vpad.right_joystick_float(x_value_float=rx, y_value_float=ry)
            self.vpad.update()
            self.v_lx, self.v_ly, self.v_rx, self.v_ry = lx, ly, rx, ry
        except Exception as e:
            self.state = ForwardState.OFF
            self.status_lbl.config(text=f"Forward: OFF (lỗi gửi: {e})", fg="#c33")
            self.toggle_btn.config(text="Forward ON")

    # ---- vòng lặp UI ----
    def tick(self):
        # Real
        r_lx, r_ly, r_rx, r_ry = self.read_axes_real()

        # UI Real
        self.lbl_lx_real.config(text=f"LX: {r_lx:+.3f}")
        self.lbl_ly_real.config(text=f"LY: {r_ly:+.3f}")
        self.lbl_rx_real.config(text=f"RX: {r_rx:+.3f}")
        self.lbl_ry_real.config(text=f"RY: {r_ry:+.3f}")

        # UI Virtual
        self.lbl_lx_virt.config(text=f"LX: {self.v_lx:+.3f}")
        self.lbl_ly_virt.config(text=f"LY: {self.v_ly:+.3f}")
        self.lbl_rx_virt.config(text=f"RX: {self.v_rx:+.3f}")
        self.lbl_ry_virt.config(text=f"RY: {self.v_ry:+.3f}")

        # State machine
        if self.state == ForwardState.ARMING:
            ok = (abs(r_lx - self.v_lx) <= ARM_EPS and
                  abs(r_ly - self.v_ly) <= ARM_EPS and
                  abs(r_rx - self.v_rx) <= ARM_EPS and
                  abs(r_ry - self.v_ry) <= ARM_EPS)
            if ok:
                self.state = ForwardState.ON
                self.status_lbl.config(text="Forward: ON", fg="#0a0")
                self.toggle_btn.config(text="Forward OFF")
                self.send_to_virtual(r_lx, r_ly, r_rx, r_ry)
        elif self.state == ForwardState.ON:
            self.send_to_virtual(r_lx, r_ly, r_rx, r_ry)

        self.root.after(POLL_MS, self.tick)

    def on_close(self):
        try:
            self.js.quit()
            pygame.quit()
        except Exception:
            pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
