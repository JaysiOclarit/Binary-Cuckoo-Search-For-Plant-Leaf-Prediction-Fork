"""
Master Explainer: Binary Cuckoo Search (BCS) Baseline Paper for Plant Leaf Feature Selection
Unified Full-Length Presentation Scene (Full Pipeline)

Combines:
- Part 1: Problem Intro & LaTeX Mathematical Formulation
- Part 2: Lévy Flight Exploration Dynamics (Mantegna Algorithm)
- Part 3: V2 = |tan(x)| Discretization Transfer Function
- Part 4: Complete Feature Selection Workflow & Convergence

Render instructions:
    # Low-res quick preview (480p, 15fps):
    manim -pql master_baseline_explainer.py MasterBaselineExplainerScene

    # High-definition (1080p, 60fps):
    manim -pqh master_baseline_explainer.py MasterBaselineExplainerScene

    # Ultra-high definition (4K, 60fps):
    manim -pqk master_baseline_explainer.py MasterBaselineExplainerScene
"""

import math
import numpy as np
from manim import *


class MasterBaselineExplainerScene(Scene):
    def construct(self):
        self.camera.background_color = "#0b0d14"

        # =====================================================================
        # CHAPTER 1: TITLE & PROBLEM CONTEXT
        # =====================================================================
        badge = RoundedRectangle(
            height=0.6, width=4.2, corner_radius=0.1,
            color=BLUE_E, fill_color="#1a1d2e", fill_opacity=0.8
        )
        badge_txt = Text("THESIS BASELINE ALGORITHM", font_size=14, weight=BOLD, color=BLUE_B).move_to(badge)
        badge_grp = VGroup(badge, badge_txt).to_edge(UP, buff=0.8)

        main_title = Text("Binary Cuckoo Search (BCS)", font_size=42, weight=BOLD, color=WHITE)
        subtitle = Text("Wrapper Feature Selection for Plant Leaf Classification", font_size=22, color=TEAL_B)
        title_block = VGroup(badge_grp, main_title, subtitle).arrange(DOWN, buff=0.25).move_to(ORIGIN)

        self.play(FadeIn(title_block, shift=UP * 0.3), run_time=1.2)
        self.wait(1.5)
        self.play(title_block.animate.scale(0.65).to_corner(UL, buff=0.5), run_time=0.9)

        # Context Cards (Leaf Features -> Curse of Dimensionality -> Binary Mask)
        card_bg = RoundedRectangle(height=2.2, width=9.0, corner_radius=0.15, color=GRAY_D, fill_color="#131622", fill_opacity=0.8)
        card_txt1 = Text("Raw Leaf Image Features: Texture, Morphology, Color Histograms (D >> 100)", font_size=18, color=WHITE)
        card_txt2 = Text("Goal: Remove redundant/noisy features to maximize classifier accuracy.", font_size=16, color=LIGHT_GRAY)
        card_content = VGroup(card_txt1, card_txt2).arrange(DOWN, buff=0.2).move_to(card_bg.get_center())
        intro_card = VGroup(card_bg, card_content).move_to(ORIGIN + DOWN * 0.5)

        self.play(Create(card_bg), FadeIn(card_content, shift=UP * 0.1), run_time=1.0)
        self.wait(1.5)
        self.play(FadeOut(intro_card), run_time=0.6)

        # =====================================================================
        # CHAPTER 2: LATEX FORMULATION & MULTI-OBJECTIVE FITNESS
        # =====================================================================
        sec_header = Text("Part 1: Mathematical Fitness Formulation", font_size=24, weight=BOLD, color=YELLOW_C)
        sec_header.to_edge(UP, buff=1.4).to_edge(LEFT, buff=0.8)
        self.play(Write(sec_header), run_time=0.6)

        # 1. Binary Solution Vector
        sol_vec = MathTex(
            r"\vec{X} = \Big[ x_1, x_2, \dots, x_D \Big] \in \{0, 1\}^D",
            font_size=32,
            color=BLUE_B
        ).move_to(ORIGIN + UP * 1.0)

        # 2. General Multi-Objective Form
        f_general = MathTex(
            r"f(\vec{X}) = w_1 \cdot \text{Accuracy}_{k\text{-NN}}(\vec{X}) + w_2 \cdot \left(1 - \frac{|\vec{X}|}{D}\right)",
            font_size=32
        ).move_to(ORIGIN + DOWN * 0.1)
        f_general.set_color_by_tex(r"\text{Accuracy}", GREEN_B)
        f_general.set_color_by_tex(r"\frac{|\vec{X}|}{D}", PURPLE_B)

        self.play(Write(sol_vec), run_time=0.9)
        self.play(Write(f_general), run_time=1.1)
        self.wait(1.0)

        # 3. Morphing to Exact Baseline Equation
        f_exact = MathTex(
            r"\max_{\vec{X}} f(\vec{X}) = \overline{\text{Accuracy}}(\vec{X}) + 0.001 \cdot \left(1 - \frac{|S|}{D}\right)",
            font_size=34
        ).move_to(ORIGIN + DOWN * 0.1)
        f_exact.set_color_by_tex(r"\max_{\vec{X}} f(\vec{X})", GOLD_B)
        f_exact.set_color_by_tex(r"\overline{\text{Accuracy}}", GREEN_B)
        f_exact.set_color_by_tex(r"0.001", TEAL_B)
        f_exact.set_color_by_tex(r"\frac{|S|}{D}", PINK)

        f_box = SurroundingRectangle(f_exact, color=YELLOW_B, buff=0.2, corner_radius=0.1)

        self.play(
            FadeTransform(f_general, f_exact),
            Create(f_box),
            run_time=1.2
        )
        self.wait(2.0)

        # Clear chapter 2
        self.play(
            FadeOut(sec_header),
            FadeOut(sol_vec),
            FadeOut(f_exact),
            FadeOut(f_box),
            run_time=0.7
        )

        # =====================================================================
        # CHAPTER 3: LÉVY FLIGHT EXPLORATION DYNAMICS
        # =====================================================================
        sec2_header = Text("Part 2: Lévy Flight Search via Mantegna's Algorithm", font_size=24, weight=BOLD, color=YELLOW_C)
        sec2_header.to_edge(UP, buff=1.4).to_edge(LEFT, buff=0.8)
        self.play(Write(sec2_header), run_time=0.6)

        # Search Plane
        grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=7.5,
            y_length=4.0,
            background_line_style={"stroke_color": GRAY_D, "stroke_width": 1, "stroke_opacity": 0.4},
            axis_config={"stroke_color": GRAY_C, "stroke_width": 1.5, "include_numbers": False}
        ).shift(DOWN * 0.7 + LEFT * 1.5)

        landscape_note = Text("Search Landscape (Feature Combinations)", font_size=15, color=GRAY_B).next_to(grid, UP, buff=0.15)

        # Formula Card on right
        info_card = RoundedRectangle(
            height=4.0, width=4.5, corner_radius=0.12,
            color=BLUE_E, fill_color="#12141f", fill_opacity=0.85
        ).to_edge(RIGHT, buff=0.6).shift(DOWN * 0.7)

        info_title = Text("Lévy Step Properties", font_size=17, weight=BOLD, color=GOLD_B).next_to(info_card.get_top(), DOWN, buff=0.2)
        info_eq1 = MathTex(r"L(s) \sim |s|^{-\lambda}", font_size=22, color=TEAL_C)
        info_eq2 = MathTex(r"\text{step} = \frac{u}{|v|^{1/\beta}}", font_size=22, color=YELLOW_B)
        info_desc = Text("• Dense local walk\n• Heavy-tailed leaps\n• Escapes local minima", font_size=14, color=WHITE, line_spacing=1.3)
        
        info_group = VGroup(info_title, info_eq1, info_eq2, info_desc).arrange(DOWN, buff=0.25).move_to(info_card.get_center())

        self.play(
            Create(grid), FadeIn(landscape_note),
            Create(info_card), FadeIn(info_group),
            run_time=1.1
        )

        # Generate sample Lévy walk with logical coordinate tracking
        np.random.seed(314)
        c_grid_coord = np.array([0.0, 0.0, 0.0])
        c_pos = grid.c2p(c_grid_coord[0], c_grid_coord[1])
        cuckoo_dot = Dot(point=c_pos, radius=0.09, color=GREEN_A)
        self.play(FadeIn(cuckoo_dot), run_time=0.4)

        # Perform 16 animated steps using logical coordinates converted via c2p
        curr_pt = c_pos
        for step_idx in range(16):
            # Mantegna-like step
            u = np.random.normal(0, 1.0)
            v = np.random.normal(0, 1.0)
            step_len = (u / (abs(v) ** (1 / 1.5))) * 0.55
            step_len = np.clip(step_len, -3.2, 3.2)
            ang = np.random.uniform(0, 2 * np.pi)
            delta = np.array([step_len * np.cos(ang), step_len * np.sin(ang), 0])
            
            # Update and clamp logical grid coordinates within grid domain [-3.6, 3.6] x [-2.6, 2.6]
            c_grid_coord = np.clip(c_grid_coord + delta, [-3.6, -2.6, 0], [3.6, 2.6, 0])
            next_pt = grid.c2p(c_grid_coord[0], c_grid_coord[1])

            is_long_leap = abs(step_len) > 1.2
            line_col = GOLD_A if is_long_leap else TEAL_C
            line_w = 3.5 if is_long_leap else 1.8

            step_line = Line(curr_pt, next_pt, stroke_width=line_w, color=line_col)

            step_anims = [
                Create(step_line),
                cuckoo_dot.animate.move_to(next_pt)
            ]

            if is_long_leap:
                pulse = Circle(radius=0.3, color=GOLD_A, stroke_width=2).move_to(next_pt)
                self.play(
                    AnimationGroup(*step_anims, FadeOut(pulse, scale=1.6), lag_ratio=0.0),
                    run_time=0.22,
                    rate_func=linear
                )
            else:
                self.play(
                    AnimationGroup(*step_anims, lag_ratio=0.0),
                    run_time=0.16,
                    rate_func=linear
                )

            curr_pt = next_pt

        self.wait(1.5)

        # Clear chapter 3
        self.play(
            FadeOut(sec2_header),
            FadeOut(grid),
            FadeOut(landscape_note),
            FadeOut(info_card),
            FadeOut(info_group),
            FadeOut(cuckoo_dot),
            *[FadeOut(mob) for mob in self.mobjects if isinstance(mob, Line)],
            run_time=0.8
        )

        # =====================================================================
        # CHAPTER 4: V-SHAPED TRANSFER FUNCTION (V2 = |tan(x)|)
        # =====================================================================
        sec3_header = Text("Part 3: Discretization with V-Shaped Transfer Function", font_size=24, weight=BOLD, color=YELLOW_C)
        sec3_header.to_edge(UP, buff=1.4).to_edge(LEFT, buff=0.8)
        self.play(Write(sec3_header), run_time=0.6)

        # Transfer function curve
        tf_axes = Axes(
            x_range=[-1.3, 1.3, 0.5],
            y_range=[0, 1.8, 0.5],
            x_length=7.0,
            y_length=3.8,
            axis_config={"color": GRAY_C, "include_numbers": False, "stroke_width": 2},
            tips=True
        ).shift(DOWN * 0.8 + LEFT * 1.6)

        tf_x_label = MathTex(r"x \text{ (Continuous Velocity / Coordinate)}", font_size=18, color=LIGHT_GRAY).next_to(tf_axes.x_axis, DOWN, buff=0.3)
        tf_y_label = MathTex(r"V_2(x) = |\tan(x)|", font_size=22, color=YELLOW_C).next_to(tf_axes.y_axis, UP, buff=0.2)

        # Threshold Line (y = 0.5)
        tf_thresh = DashedLine(
            start=tf_axes.c2p(-1.25, 0.5),
            end=tf_axes.c2p(1.25, 0.5),
            dash_length=0.1,
            color=YELLOW_B,
            stroke_width=2.5
        )
        tf_thresh_txt = MathTex(r"\tau = 0.5", font_size=20, color=YELLOW_B).next_to(tf_thresh, UP, buff=0.08).to_edge(LEFT, buff=1.0)

        # Plot bounded to [-1.0, 1.0] where max y = tan(1.0) approx 1.557, inside y_range [0, 1.8]
        tf_curve = tf_axes.plot(
            lambda x: np.abs(np.tan(x)),
            x_range=[-1.0, 1.0, 0.01],
            color=TEAL_B,
            stroke_width=4
        )

        # Right-side Decision Card
        tf_panel = RoundedRectangle(
            height=3.8, width=4.6, corner_radius=0.12,
            color=BLUE_E, fill_color="#12141f", fill_opacity=0.85
        ).to_edge(RIGHT, buff=0.6).shift(DOWN * 0.8)

        tf_panel_title = Text("Binary Feature Rule", font_size=18, weight=BOLD, color=GOLD_B).next_to(tf_panel.get_top(), DOWN, buff=0.2)
        tf_rule = MathTex(
            r"X_j = \begin{cases} 1 & V_2(x) \ge 0.5 \\ 0 & V_2(x) < 0.5 \end{cases}",
            font_size=24
        ).next_to(tf_panel_title, DOWN, buff=0.3)
        tf_rule.set_color_by_tex(r"1", GREEN_B)
        tf_rule.set_color_by_tex(r"0", RED_B)

        tf_note = Text("Symmetric V-shape ensures\nhigh-magnitude steps toggle/activate\nfeatures without saturation.", font_size=13, color=LIGHT_GRAY, line_spacing=1.2)
        tf_note.next_to(tf_rule, DOWN, buff=0.3)

        self.play(
            Create(tf_axes), Write(tf_x_label), Write(tf_y_label),
            Create(tf_thresh), Write(tf_thresh_txt),
            Create(tf_curve),
            Create(tf_panel), Write(tf_panel_title), Write(tf_rule), FadeIn(tf_note),
            run_time=1.3
        )
        self.wait(2.5)

        # Clear chapter 4
        self.play(
            FadeOut(sec3_header),
            FadeOut(tf_axes),
            FadeOut(tf_x_label),
            FadeOut(tf_y_label),
            FadeOut(tf_thresh),
            FadeOut(tf_thresh_txt),
            FadeOut(tf_curve),
            FadeOut(tf_panel),
            FadeOut(tf_panel_title),
            FadeOut(tf_rule),
            FadeOut(tf_note),
            run_time=0.8
        )

        # =====================================================================
        # CHAPTER 5: SUMMARY & CONCLUSION
        # =====================================================================
        summary_card = RoundedRectangle(
            height=3.8, width=10.5, corner_radius=0.15,
            color=GOLD_B, fill_color="#12141f", fill_opacity=0.95
        ).move_to(ORIGIN + DOWN * 0.2)

        sum_title = Text("Summary: Why Binary Cuckoo Search Excels for Leaf Classification", font_size=20, weight=BOLD, color=GOLD_A)
        sum_p1 = Text("1. Lévy Flights provide superior global exploration compared to standard random walks.", font_size=16, color=WHITE)
        sum_p2 = Text("2. V2 = |tan(x)| transfer function maintains balanced exploration & exploitation.", font_size=16, color=WHITE)
        sum_p3 = Text("3. Compact feature subsets achieve up to 99.6% accuracy on benchmark leaf datasets.", font_size=16, color=GREEN_C)

        sum_group = VGroup(sum_title, sum_p1, sum_p2, sum_p3).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to(summary_card.get_center())

        self.play(Create(summary_card), FadeIn(sum_group, shift=UP * 0.15), run_time=1.2)
        self.wait(3.5)

        # Outro
        self.play(FadeOut(summary_card), FadeOut(sum_group), FadeOut(title_block), run_time=1.0)
        
        thanks = Text("End of Math Engine Scene", font_size=32, weight=BOLD, color=BLUE_B).move_to(ORIGIN)
        self.play(FadeIn(thanks, scale=0.9), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(thanks), run_time=0.8)
