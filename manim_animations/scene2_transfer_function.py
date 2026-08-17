"""
Scene 2: V-Shaped Transfer Function V2 = |tan(x)|
Visualizes:
1. The mathematical curve of V2(x) = |tan(x)|
2. The threshold boundary at y = 0.5
3. Dynamic probe tracing x -> V2(x) -> Discrete Binary Output {0, 1}
4. Comparison with S-shaped functions & why V-shaped is used in BCS

Run with:
    manim -pql scene2_transfer_function.py TransferFunctionScene
For high quality (1080p60):
    manim -pqh scene2_transfer_function.py TransferFunctionScene
"""

import numpy as np
from manim import *


class TransferFunctionScene(Scene):
    def construct(self):
        # Configure dark aesthetic theme
        self.camera.background_color = "#0f111a"

        # ---------------------------------------------------------------------
        # 1. Header Banner
        # ---------------------------------------------------------------------
        title = Text("Discretization Engine: V-Shaped Transfer Function", font_size=32, weight=BOLD, color=BLUE_C)
        formula = MathTex(r"V_2(x) = |\tan(x)|", font_size=36, color=YELLOW_C)
        header = VGroup(title, formula).arrange(DOWN, buff=0.15).to_edge(UP, buff=0.4)

        self.play(FadeIn(header, shift=DOWN * 0.2), run_time=0.9)
        self.wait(0.5)

        # ---------------------------------------------------------------------
        # 2. Coordinate System
        # ---------------------------------------------------------------------
        # We plot in range [-1.0, 1.0] to keep |tan(x)| <= 1.56, comfortably within y_range [0, 1.8]
        axes = Axes(
            x_range=[-1.3, 1.3, 0.5],
            y_range=[0, 1.8, 0.5],
            x_length=7.5,
            y_length=4.2,
            axis_config={"color": GRAY_C, "include_numbers": False, "stroke_width": 2},
            tips=True,
        ).shift(DOWN * 0.6 + LEFT * 1.5)

        # Custom Axis Labels
        x_labels = VGroup(
            MathTex(r"-1.0", font_size=18).next_to(axes.c2p(-1.0, 0), DOWN, buff=0.15),
            MathTex(r"-0.5", font_size=18).next_to(axes.c2p(-0.5, 0), DOWN, buff=0.15),
            MathTex(r"0", font_size=18).next_to(axes.c2p(0, 0), DOWN + LEFT * 0.1, buff=0.15),
            MathTex(r"+0.5", font_size=18).next_to(axes.c2p(0.5, 0), DOWN, buff=0.15),
            MathTex(r"+1.0", font_size=18).next_to(axes.c2p(1.0, 0), DOWN, buff=0.15),
            MathTex(r"x \text{ (Continuous Velocity/Step)}", font_size=20, color=LIGHT_GRAY).next_to(axes.x_axis, DOWN, buff=0.4)
        )

        y_labels = VGroup(
            MathTex(r"0.0", font_size=18).next_to(axes.c2p(0, 0), LEFT, buff=0.15),
            MathTex(r"0.5", font_size=18, color=YELLOW_B).next_to(axes.c2p(0, 0.5), LEFT, buff=0.15),
            MathTex(r"1.0", font_size=18).next_to(axes.c2p(0, 1.0), LEFT, buff=0.15),
            MathTex(r"1.5", font_size=18).next_to(axes.c2p(0, 1.5), LEFT, buff=0.15),
            MathTex(r"V_2(x)", font_size=22, color=YELLOW_C).next_to(axes.y_axis, UP, buff=0.2)
        )

        self.play(Create(axes), Write(x_labels), Write(y_labels), run_time=1.2)

        # ---------------------------------------------------------------------
        # 3. Threshold Line (y = 0.5)
        # ---------------------------------------------------------------------
        thresh_line = DashedLine(
            start=axes.c2p(-1.25, 0.5),
            end=axes.c2p(1.25, 0.5),
            dash_length=0.1,
            color=YELLOW_B,
            stroke_width=2.5
        )
        thresh_label = MathTex(r"\text{Threshold } \tau = 0.5", font_size=20, color=YELLOW_B)
        thresh_label.next_to(thresh_line, UP, buff=0.08).to_edge(LEFT, buff=0.8)

        self.play(Create(thresh_line), FadeIn(thresh_label), run_time=0.8)

        # ---------------------------------------------------------------------
        # 4. Plotting V2 Curve
        # ---------------------------------------------------------------------
        def v2_func(x):
            return np.abs(np.tan(x))

        # Plot across [-1.0, 1.0] where max value is tan(1.0) approx 1.557
        v2_curve = axes.plot(
            v2_func,
            x_range=[-1.0, 1.0, 0.01],
            color=TEAL_C,
            stroke_width=4
        )

        curve_tag = MathTex(r"V_2(x) = |\tan(x)|", font_size=22, color=TEAL_C)
        curve_tag.next_to(axes.c2p(0.85, v2_func(0.85)), UR, buff=0.15)

        self.play(Create(v2_curve), Write(curve_tag), run_time=1.5)
        self.wait(0.5)

        # ---------------------------------------------------------------------
        # 5. Right-Hand Side Info & Logic Panel
        # ---------------------------------------------------------------------
        panel_bg = RoundedRectangle(
            height=4.6, width=4.8, corner_radius=0.15,
            color=BLUE_E, fill_color="#141724", fill_opacity=0.9
        ).to_edge(RIGHT, buff=0.6).shift(DOWN * 0.4)

        panel_title = Text("Discretization Rule", font_size=20, weight=BOLD, color=GOLD_B)
        panel_title.next_to(panel_bg.get_top(), DOWN, buff=0.2)

        rule_tex = MathTex(
            r"X_j^{(t+1)} = \begin{cases} 1 & \text{if } V_2(x) \ge 0.5 \\ 0 & \text{if } V_2(x) < 0.5 \end{cases}",
            font_size=24
        )
        rule_tex.set_color_by_tex(r"1", GREEN_B)
        rule_tex.set_color_by_tex(r"0", RED_B)
        rule_tex.next_to(panel_title, DOWN, buff=0.3)

        # Interactive Probe Display Box
        probe_box = RoundedRectangle(
            height=1.8, width=4.2, corner_radius=0.1,
            color=GRAY_D, fill_color="#0a0c13", fill_opacity=0.8
        ).next_to(rule_tex, DOWN, buff=0.25)

        x_val_tex = MathTex(r"x = ", r"+0.00", font_size=22, color=WHITE)
        v2_val_tex = MathTex(r"V_2(x) = ", r"0.00", font_size=22, color=YELLOW_B)
        bit_val_tex = MathTex(r"\text{Feature Bit } X_j = ", r"0 \text{ (Deselected)}", font_size=22, color=RED_B)

        probe_group = VGroup(x_val_tex, v2_val_tex, bit_val_tex).arrange(DOWN, buff=0.12).move_to(probe_box.get_center())

        self.play(
            Create(panel_bg),
            Write(panel_title),
            Write(rule_tex),
            Create(probe_box),
            FadeIn(probe_group),
            run_time=1.2
        )

        # ---------------------------------------------------------------------
        # 6. Dynamic Scanning Point (ValueTracker Animation)
        # ---------------------------------------------------------------------
        x_tracker = ValueTracker(-0.95)

        # Glowing dot on curve
        dot = always_redraw(
            lambda: Dot(
                point=axes.c2p(x_tracker.get_value(), v2_func(x_tracker.get_value())),
                radius=0.1,
                color=GREEN_A if v2_func(x_tracker.get_value()) >= 0.5 else RED_A
            )
        )

        # Projection dashed lines to axes
        proj_x = always_redraw(
            lambda: DashedLine(
                start=axes.c2p(x_tracker.get_value(), 0),
                end=axes.c2p(x_tracker.get_value(), v2_func(x_tracker.get_value())),
                color=GRAY_B,
                stroke_width=1.5
            )
        )

        # Live text updaters
        def update_probe_texts():
            xv = x_tracker.get_value()
            yv = v2_func(xv)
            is_active = yv >= 0.5

            new_x = MathTex(r"x = ", f"{xv:+.2f}", font_size=22, color=WHITE)
            new_v2 = MathTex(r"V_2(x) = ", f"{yv:.2f}", font_size=22, color=YELLOW_B)
            
            if is_active:
                new_bit = MathTex(r"\text{Feature Bit } X_j = ", r"1 \text{ (Selected)}", font_size=22, color=GREEN_B)
            else:
                new_bit = MathTex(r"\text{Feature Bit } X_j = ", r"0 \text{ (Deselected)}", font_size=22, color=RED_B)

            new_grp = VGroup(new_x, new_v2, new_bit).arrange(DOWN, buff=0.12).move_to(probe_box.get_center())
            return new_grp

        live_probe_group = always_redraw(update_probe_texts)

        self.play(
            FadeIn(dot),
            Create(proj_x),
            FadeOut(probe_group),
            FadeIn(live_probe_group),
            run_time=0.6
        )

        # Animate probe moving across the domain from -0.95 to +0.95
        self.play(x_tracker.animate.set_value(0.0), run_time=2.2, rate_func=linear)
        self.wait(0.4)
        self.play(x_tracker.animate.set_value(0.95), run_time=2.2, rate_func=linear)
        self.wait(1.0)

        # ---------------------------------------------------------------------
        # 7. Symmetry & Saturation Avoidance Note
        # ---------------------------------------------------------------------
        callout = Text(
            "V-shaped functions preserve exploration:\nHigh step magnitude (|x| > 0.46) activates features regardless of sign.",
            font_size=18,
            color=LIGHT_PINK,
            line_spacing=1.2
        ).to_edge(DOWN, buff=0.25).shift(LEFT * 0.8)

        self.play(FadeIn(callout, shift=UP * 0.2), run_time=1.0)
        self.wait(2.5)

        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.0)
