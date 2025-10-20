import sys
import pygame

# Mapping cố định theo yêu cầu:
AX_LX = 0  # A00
AX_LY = 1  # A01
AX_RY = 2  # A02
AX_RX = 3  # A03

DEADZONE = 0.08  # đặt 0.0 nếu muốn thấy cả nhiễu nhỏ

def dz(v, d=DEADZONE):
    return 0.0 if abs(v) < d else v

def main():
    pygame.init()
    pygame.joystick.init()

    # Tạo cửa sổ nhỏ, chữ gọn
    screen = pygame.display.set_mode((600, 180))
    pygame.display.set_caption("RX RY LX LY Viewer")
    font_name = "consolas"
    font_small = pygame.font.SysFont(font_name, 18)
    font_val   = pygame.font.SysFont(font_name, 22, bold=True)

    # Kiểm tra tay cầm
    if pygame.joystick.get_count() == 0:
        print("⚠️  Không tìm thấy tay cầm. Hãy cắm tay cầm rồi chạy lại.")
        pygame.quit()
        sys.exit(1)

    js = pygame.joystick.Joystick(0)
    js.init()
    pad_name = js.get_name()
    print(f"Đang đọc từ tay cầm: {pad_name}")

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        pygame.event.pump()

        # Đọc 4 trục theo mapping
        try:
            rx = dz(js.get_axis(AX_RX))
            ry = dz(js.get_axis(AX_RY))
            lx = dz(js.get_axis(AX_LX))
            ly = dz(js.get_axis(AX_LY))
        except Exception:
            rx = ry = lx = ly = 0.0

        # Vẽ tối giản
        screen.fill((0, 0, 0))
        # (Tùy chọn) Hiển thị tên tay cầm để tiện kiểm tra
        screen.blit(font_small.render(f"Tay cầm: {pad_name}   (Esc để thoát)", True, (200,200,200)), (20, 18))

        # Chỉ hiển thị 4 giá trị RX RY LX LY
        screen.blit(font_val.render(f"RX: {rx:+.3f}", True, (0,255,0)), (20, 60))
        screen.blit(font_val.render(f"RY: {ry:+.3f}", True, (0,255,0)), (20, 90))
        screen.blit(font_val.render(f"LX: {lx:+.3f}", True, (0,255,0)), (260, 60))
        screen.blit(font_val.render(f"LY: {ly:+.3f}", True, (0,255,0)), (260, 90))

        pygame.display.flip()
        clock.tick(60)

    js.quit()
    pygame.quit()

if __name__ == "__main__":
    main()
