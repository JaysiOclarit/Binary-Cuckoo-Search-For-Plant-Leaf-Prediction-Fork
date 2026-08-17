"""
Scene 3: Simulating Exact Lévy Flight Random Walk on a 2D Grid
Visualizes:
1. Mantegna's Algorithm for exact Lévy flight generation (lambda = 1.5)
2. Side-by-side comparison: Standard Brownian Motion vs Lévy Flight
3. Heavy-tailed jump mechanism enabling escape from local optima in feature selection

Run with:
    manim -pql scene3_levy_flight_walk.py LevyFlightScene
For high quality (1080p60):
    manim -pqh scene3_levy_flight_walk.py LevyFlightScene
"""

import math
import numpy as np
from manim import *


def generate_levy_flight_steps(num_steps=35, beta=1.5, scale=0.45, seed=42):
    """
    Mantegna's algorithm for simulating exact Lévy flight step lengths
    """
    np.random.seed(seed)
    
    # Calculate Mantegna's sigma_u
    numerator = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    denominator = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (numerator / denominator) ** (1 / beta)
    sigma_v = 1.0

    steps = []
    for _ in range(num_steps):
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, sigma_v)
        step_len = (u / (abs(v) ** (1 / beta))) * scale
        
        # Clip excessive outlier jumps to keep visually inside grid bounds
        step_len = np.clip(step_len, -2.8, 2.8)
        
        # Random direction in 2D
        angle = np.random.uniform(0, 2 * np.pi)
        dx = step_len * np.cos(angle)
        dy = step_len * np.sin(angle)
        steps.append(np.array([dx, dy, 0]))
        
    return steps


def generate_brownian_steps(num_steps=35, scale=0.25, seed=42):
    """
    Gaussian random walk (Standard Brownian Motion)
    """
    np.random.seed(seed)
    steps = []
    for _ in range(num_steps):
        dx = np.random.normal(0, scale)
        dy = np.random.normal(0, scale)
        steps.append(np.array([dx, dy, 0]))
    return steps


class LevyFlightScene(Scene):
    def construct(self):
        # Configure dark aesthetic theme
        self.camera.background_color = "#0f111a"

        # ---------------------------------------------------------------------
        # 1. Title Banner & Math Intro
        # ---------------------------------------------------------------------
        title = Text("Exploration Engine: Lévy Flight Simulation", font_size=32, weight=BOLD, color=BLUE_C)
        levy_law = MathTex(
            r"\text{L\'evy}(\lambda) \sim u = t^{-\lambda}, \quad 1 < \lambda \le 3 \quad (\text{Scale-Free Superdiffusion})",
            font_size=24,
            color=YELLOW_C
        )
        header = VGroup(title, levy_law).arrange(DOWN, buff=0.15).to_edge(UP, buff=0.4)

        self.play(FadeIn(header, shift=DOWN * 0.2), run_time=0.9)
        self.wait(0.5)

        # ---------------------------------------------------------------------
        # 2. Dual Grids Setup (Brownian vs Lévy Flight)
        # ---------------------------------------------------------------------
        # Left Grid: Brownian Motion
        left_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5.0,
            y_length=4.2,
            background_line_style={"stroke_color": GRAY_D, "stroke_width": 1, "stroke_opacity": 0.4},
            axis_config={"stroke_color": GRAY_C, "stroke_width": 1.5, "include_numbers": False}
        ).shift(LEFT * 3.4 + DOWN * 0.7)

        left_box = RoundedRectangle(
            height=4.8, width=5.6, corner_radius=0.12,
            color=GRAY_E, fill_color="#12141f", fill_opacity=0.5
        ).move_to(left_grid.get_center())

        left_label = Text("Standard Brownian Walk", font_size=18, weight=BOLD, color=RED_C)
        left_label.next_to(left_box.get_top(), DOWN, buff=0.15)
        left_sub = Text("Gaussian Step: Trapped in Local Optima", font_size=13, color=GRAY_B)
        left_sub.next_to(left_label, DOWN, buff=0.08)

        # Right Grid: Lévy Flight
        right_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5.0,
            y_length=4.2,
            background_line_style={"stroke_color": GRAY_D, "stroke_width": 1, "stroke_opacity": 0.4},
            axis_config={"stroke_color": GRAY_C, "stroke_width": 1.5, "include_numbers": False}
        ).shift(RIGHT * 3.4 + DOWN * 0.7)

        right_box = RoundedRectangle(
            height=4.8, width=5.6, corner_radius=0.12,
            color=BLUE_E, fill_color="#12141f", fill_opacity=0.5
        ).move_to(right_grid.get_center())

        right_label = Text("Lévy Flight Walk (Mantegna's)", font_size=18, weight=BOLD, color=GREEN_C)
        right_label.next_to(right_box.get_top(), DOWN, buff=0.15)
        right_sub = Text("Heavy-Tailed Jumps: Escapes Local Optima", font_size=13, color=GRAY_B)
        right_sub.next_to(right_label, DOWN, buff=0.08)

        self.play(
            Create(left_box), Create(left_grid), Write(left_label), FadeIn(left_sub),
            Create(right_box), Create(right_grid), Write(right_label), FadeIn(right_sub),
            run_time=1.2
        )
        self.wait(0.5)

        # ---------------------------------------------------------------------
        # 3. Simulate Walks (Logical Grid Coordinates -> c2p Conversion)
        # ---------------------------------------------------------------------
        b_steps = generate_brownian_steps(num_steps=30, scale=0.35, seed=101)
        l_steps = generate_levy_flight_steps(num_steps=30, beta=1.5, scale=0.75, seed=202)

        # Track positions in logical coordinate space [-3, 3]
        b_grid_coord = np.array([0.0, 0.0, 0.0])
        l_grid_coord = np.array([0.0, 0.0, 0.0])

        b_pos = left_grid.c2p(b_grid_coord[0], b_grid_coord[1])
        l_pos = right_grid.c2p(l_grid_coord[0], l_grid_coord[1])

        b_dot = Dot(point=b_pos, radius=0.08, color=RED_B)
        l_dot = Dot(point=l_pos, radius=0.08, color=GREEN_B)

        self.play(FadeIn(b_dot), FadeIn(l_dot), run_time=0.5)

        # Animate trajectory steps sequentially
        b_current = b_pos
        l_current = l_pos

        for i in range(25):
            # Update logical grid positions and clamp within grid boundaries [-2.7, 2.7]
            b_grid_coord = np.clip(b_grid_coord + b_steps[i], [-2.7, -2.7, 0.0], [2.7, 2.7, 0.0])
            l_grid_coord = np.clip(l_grid_coord + l_steps[i], [-2.7, -2.7, 0.0], [2.7, 2.7, 0.0])

            # Convert logical coordinates to actual screen coordinates via c2p
            b_next = left_grid.c2p(b_grid_coord[0], b_grid_coord[1])
            l_next = right_grid.c2p(l_grid_coord[0], l_grid_coord[1])

            # Line segments
            b_line = Line(b_current, b_next, stroke_width=2, color=RED_D, stroke_opacity=0.8)
            
            # Highlight long jumps in Lévy flight with Gold/Yellow
            step_mag = np.linalg.norm(l_steps[i])
            if step_mag > 1.1:
                l_color = GOLD_A
                l_width = 3.5
            else:
                l_color = TEAL_C
                l_width = 2.0

            l_line = Line(l_current, l_next, stroke_width=l_width, color=l_color)

            t_run = 0.14 if i < 18 else 0.20
            
            step_animations = [
                Create(b_line),
                b_dot.animate.move_to(b_next),
                Create(l_line),
                l_dot.animate.move_to(l_next)
            ]

            # Flash when a heavy-tailed jump occurs
            if step_mag > 1.5:
                flash = Circle(radius=0.25, color=YELLOW_A, stroke_width=2).move_to(l_next)
                self.play(
                    AnimationGroup(*step_animations, FadeOut(flash, scale=1.8), lag_ratio=0.0),
                    run_time=t_run + 0.1,
                    rate_func=linear
                )
            else:
                self.play(
                    AnimationGroup(*step_animations, lag_ratio=0.0),
                    run_time=t_run,
                    rate_func=linear
                )

            b_current = b_next
            l_current = l_next

        self.wait(0.8)

        # ---------------------------------------------------------------------
        # 4. Mantegna Formulation Inset Card
        # ---------------------------------------------------------------------
        mantegna_box = RoundedRectangle(
            height=1.7, width=10.5, corner_radius=0.15,
            color=PURPLE_C, fill_color="#10121d", fill_opacity=0.95
        ).to_edge(DOWN, buff=0.35)

        mantegna_title = Text("Mantegna's Step Formulation:", font_size=17, weight=BOLD, color=GOLD_B)
        mantegna_eq = MathTex(
            r"\text{step} = \frac{u}{|v|^{1/\beta}}, \quad u \sim \mathcal{N}(0, \sigma_u^2), \quad v \sim \mathcal{N}(0, 1), \quad "
            r"\sigma_u = \left( \frac{\Gamma(1+\beta) \sin(\pi \beta / 2)}{\Gamma(\frac{1+\beta}{2}) \beta 2^{(\beta-1)/2}} \right)^{1/\beta}",
            font_size=21
        ).set_color_by_tex(r"\sigma_u", TEAL_B)

        mantegna_content = VGroup(mantegna_title, mantegna_eq).arrange(DOWN, buff=0.15).move_to(mantegna_box.get_center())

        self.play(
            Create(mantegna_box),
            FadeIn(mantegna_content, shift=UP * 0.15),
            run_time=1.0
        )
        self.wait(3.0)

        # Fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.0)
