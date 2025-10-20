import sys
import pygame

# Cấu hình hiển thị
BG = (0, 0, 0)
FG = (220, 220, 220)
ACCENT = (0, 255, 0)
LINE_H = 24
PADDING_X = 18
PADDING_Y = 16
DEADZONE = 0.08  # áp cho trục analog để lọc nhiễu nhỏ

def dz(v, dz=DEADZONE):
    return 0.0 if abs(v) < dz else v

def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("⚠️  Không tìm thấy tay cầm nào. Hãy cắm tay cầm rồi chạy lại.")
        sys.exit(1)

    js = pygame.joystick.Joystick(0)
    js.init()
    name = js.get_name()

    # Lấy số lượng phần tử
    n_axes = js.get_numaxes()
    n_buttons = js.get_numbuttons()
    n_hats = js.get_numhats()
    n_balls = js.get_numballs()

    pygame.display.set_caption("Gamepad Dumper (All values)")
    font_title = pygame.font.SysFont("consolas", 18, bold=True)
    font_line = pygame.font.SysFont("consolas", 18)

    # Tính chiều cao cửa sổ đủ để hiển thị hết
    lines = 2  # tiêu đề + dòng thống kê
    if n_axes:   lines += 1 + n_axes
    if n_buttons:lines += 1 + n_buttons
    if n_hats:   lines += 1 + n_hats
    if n_balls:  lines += 1 + n_balls
    height = max(220, PADDING_Y * 2 + lines * LINE_H)
    width = 640
    screen = pygame.display.set_mode((width, height))

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        pygame.event.pump()

        # Đọc giá trị
        axes = [dz(js.get_axis(i)) for i in range(n_axes)]
        buttons = [js.get_button(i) for i in range(n_buttons)]
        hats = [js.get_hat(i) for i in range(n_hats)]
        balls = [js.get_ball(i) for i in range(n_balls)]

        # Vẽ
        screen.fill(BG)
        x = PADDING_X
        y = PADDING_Y

        # Tiêu đề & thống kê
        screen.blit(font_title.render(f"Tay cầm: {name}", True, FG), (x, y)); y += LINE_H
        stats = f"Axes: {n_axes}   Buttons: {n_buttons}   Hats: {n_hats}   Balls: {n_balls}   (Esc thoát)"
        screen.blit(font_line.render(stats, True, (180,180,180)), (x, y)); y += LINE_H

        # Axes
        if n_axes:
            screen.blit(font_title.render("AXES:", True, ACCENT), (x, y)); y += LINE_H
            for i, v in enumerate(axes):
                screen.blit(font_line.render(f"A{i:02d}: {v:+.3f}", True, FG), (x, y)); y += LINE_H

        # Buttons
        if n_buttons:
            screen.blit(font_title.render("BUTTONS:", True, ACCENT), (x, y)); y += LINE_H
            for i, b in enumerate(buttons):
                # b là 0/1
                screen.blit(font_line.render(f"B{i:02d}: {b}", True, FG), (x, y)); y += LINE_H

        # Hats (D-Pad thường là hat 2D: (-1..1, -1..1))
        if n_hats:
            screen.blit(font_title.render("HATS:", True, ACCENT), (x, y)); y += LINE_H
            for i, (hx, hy) in enumerate(hats):
                screen.blit(font_line.render(f"H{i:02d}: ({hx:+d}, {hy:+d})", True, FG), (x, y)); y += LINE_H

        # Balls (ít gặp; chuột bi trên joystick)
        if n_balls:
            screen.blit(font_title.render("BALLS:", True, ACCENT), (x, y)); y += LINE_H
            for i, (bx, by) in enumerate(balls):
                screen.blit(font_line.render(f"Ball{i:02d}: (dx={bx:+.3f}, dy={by:+.3f})", True, FG), (x, y)); y += LINE_H

        pygame.display.flip()
        clock.tick(60)

    js.quit()
    pygame.quit()

if __name__ == "__main__":
    main()
