import math
import sys
import tkinter as tk

# vgamepad: gói Python bọc ViGEm (Xbox 360/DS4 virtual)
try:
    import vgamepad as vg
except ImportError:
    print("Chưa cài 'vgamepad'. Hãy chạy: pip install vgamepad")
    sys.exit(1)


class StickWidget:
    """
    Joystick hình tròn đơn giản: kéo núm trong phạm vi vòng tròn.
    Trả về (x, y) đã chuẩn hoá trong [-1.0, 1.0], với (0,0) là tâm.
    """
    def __init__(self, parent, title="Stick", size=220, pad=12, radius_ratio=0.7, deadzone=0.05):
        self.frame = tk.Frame(parent)
        self.title = tk.Label(self.frame, text=title, font=("Segoe UI", 11, "bold"))
        self.title.pack(pady=(0, 4))

        self.size = size
        self.canvas = tk.Canvas(self.frame, width=size, height=size, bg="#111", highlightthickness=0)
        self.canvas.pack()

        self.center = (size // 2, size // 2)
        self.radius = int((size // 2 - pad) * radius_ratio)  # biên tối đa của cần
        self.deadzone = deadzone

        # Vẽ nền/biên
        c = self.center
        R = self.radius
        self.canvas.create_oval(c[0]-R, c[1]-R, c[0]+R, c[1]+R, outline="#444", width=2)
        self.canvas.create_line(c[0], 10, c[0], size-10, fill="#333")
        self.canvas.create_line(10, c[1], size-10, c[1], fill="#333")

        # Vẽ núm (thumb)
        self.knob_r = 14
        self.knob = self.canvas.create_oval(
            c[0]-self.knob_r, c[1]-self.knob_r, c[0]+self.knob_r, c[1]+self.knob_r,
            fill="#2c7", outline="#0b4"
        )

        # Trạng thái
        self.dragging = False
        self.norm_xy = (0.0, 0.0)

        # Bind chuột
        self.canvas.tag_bind(self.knob, "<ButtonPress-1>", self._on_press)
        self.canvas.tag_bind(self.knob, "<B1-Motion>", self._on_drag)
        self.canvas.tag_bind(self.knob, "<ButtonRelease-1>", self._on_release)

        # Cho phép click nền để nhảy núm đến vị trí đó
        self.canvas.bind("<ButtonPress-1>", self._on_press_anywhere)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Nhấn chuột phải để trả về tâm
        self.canvas.bind("<Button-3>", lambda e: self.reset_to_center())

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def _on_press_anywhere(self, event):
        # Nếu bấm vào nền (không trúng knob), vẫn cho kéo
        self._on_press(event)

    def _on_press(self, event):
        self.dragging = True
        self._move_knob(event.x, event.y)

    def _on_drag(self, event):
        if self.dragging:
            self._move_knob(event.x, event.y)

    def _on_release(self, event):
        self.dragging = False
        # vẫn để nguyên vị trí; muốn auto snap về giữa thì bật dòng sau:
        # self.reset_to_center()

    def _move_knob(self, x, y):
        # Hạn chế trong vòng tròn biên
        cx, cy = self.center
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)
        if dist > self.radius:
            scale = self.radius / dist
            dx *= scale
            dy *= scale
            x = cx + dx
            y = cy + dy

        # Cập nhật hình
        r = self.knob_r
        self.canvas.coords(self.knob, x - r, y - r, x + r, y + r)

        # Chuẩn hoá sang [-1, 1] (lưu ý: trục Y ngược để khớp tiêu chuẩn gamepad)
        nx = dx / self.radius
        ny = -dy / self.radius

        # Áp deadzone
        if math.hypot(nx, ny) < self.deadzone:
            nx, ny = 0.0, 0.0

        self.norm_xy = (max(-1.0, min(1.0, nx)), max(-1.0, min(1.0, ny)))

    def reset_to_center(self):
        cx, cy = self.center
        r = self.knob_r
        self.canvas.coords(self.knob, cx - r, cy - r, cx + r, cy + r)
        self.norm_xy = (0.0, 0.0)

    def get(self):
        """Trả về (x, y) đã chuẩn hoá trong [-1.0, 1.0]."""
        return self.norm_xy


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Trungtaulua Virtual Gamepad")

        # Tạo tay cầm Xbox360 ảo
        try:
            self.gamepad = vg.VX360Gamepad()
        except Exception as e:
            tk.messagebox.showerror("Lỗi", f"Không khởi tạo được tay cầm ảo.\n{e}")
            sys.exit(2)

        # Khung UI
        wrap = tk.Frame(self.root, padx=12, pady=12, bg="#0a0a0a")
        wrap.pack(fill="both", expand=True)

        # Hai joystick
        sticks = tk.Frame(wrap, bg="#0a0a0a")
        sticks.pack()

        self.left = StickWidget(sticks, title="Left Stick (LX, LY)")
        self.right = StickWidget(sticks, title="Right Stick (RX, RY)")
        self.left.pack(side="left", padx=10)
        self.right.pack(side="left", padx=10)

        # Nút tiện ích
        btns = tk.Frame(wrap, pady=10, bg="#0a0a0a")
        btns.pack(fill="x")
        tk.Button(btns, text="Center Both (Space)", command=self.center_both).pack(side="left")
        tk.Button(btns, text="Quit (Esc)", command=self.quit).pack(side="right")

        # Phím tắt
        self.root.bind("<space>", lambda e: self.center_both())
        self.root.bind("<Escape>", lambda e: self.quit())

        # Nhãn trạng thái
        self.status = tk.StringVar(value="LX=0.00  LY=0.00   RX=0.00  RY=0.00")
        tk.Label(wrap, textvariable=self.status, fg="#ddd", bg="#0a0a0a").pack(anchor="w", pady=(6, 0))

        # Vòng cập nhật gửi trạng thái sang tay cầm ảo
        self._tick()

    def center_both(self):
        self.left.reset_to_center()
        self.right.reset_to_center()
        self._send_to_gamepad()  # gửi ngay

    def _send_to_gamepad(self):
        lx, ly = self.left.get()
        rx, ry = self.right.get()

        # Dùng API float chuẩn hoá [-1.0, 1.0]
        self.gamepad.left_joystick_float(x_value_float=lx, y_value_float=ly)
        self.gamepad.right_joystick_float(x_value_float=rx, y_value_float=ry)

        # Gửi report
        self.gamepad.update()

        self.status.set(f"LX={lx:+.2f}  LY={ly:+.2f}   RX={rx:+.2f}  RY={ry:+.2f}")

    def _tick(self):
        # gọi đều đặn để phản ánh mọi thay đổi kéo thả
        self._send_to_gamepad()
        # 60 Hz ~ 16 ms; ở đây chọn 10–15 ms cho phản hồi mượt
        self.root.after(15, self._tick)

    def quit(self):
        try:
            # Đặt về 0 trước khi thoát để tránh trục kẹt
            self.left.reset_to_center()
            self.right.reset_to_center()
            self._send_to_gamepad()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    app = App()
    app.root.mainloop()
