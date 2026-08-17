"""
Scene 1: LaTeX Mathematical Formulation & Morphing Animations
Explaining the Binary Cuckoo Search (BCS) baseline paper optimization equations:
1. Multi-Objective Fitness Function f(X)
2. Lévy Flight Exploration Update
3. Nest Abandonment / Discovery Rule

Run with:
    manim -pql scene1_latex_morphing.py LatexFormulaScene
For high quality (1080p60):
    manim -pqh scene1_latex_morphing.py LatexFormulaScene
"""

from manim import *


class LatexFormulaScene(Scene):
    def construct(self):
        # Configure dark aesthetic theme
        self.camera.background_color = "#0f111a"

        # ---------------------------------------------------------------------
        # 1. Title Banner
        # ---------------------------------------------------------------------
        title = Text("Binary Cuckoo Search (BCS)", font_size=38, weight=BOLD, color=BLUE_C)
        subtitle = Text("Baseline Paper Mathematical Formulation", font_size=20, color=GRAY_B)
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.15).to_edge(UP, buff=0.5)

        self.play(FadeIn(header, shift=DOWN * 0.3), run_time=1.0)
        self.wait(0.5)

        # ---------------------------------------------------------------------
        # 2. Candidate Solution Vector Representation
        # ---------------------------------------------------------------------
        sec1_label = Text("1. Solution Representation (Binary Feature Mask)", font_size=22, color=YELLOW_C)
        sec1_label.next_to(header, DOWN, buff=0.45).to_edge(LEFT, buff=1.0)

        vec_tex = MathTex(
            r"\vec{X}_i", r"=", r"\Big[", r"x_{i,1}", r",", r"x_{i,2}", r",", r"\dots", r",", r"x_{i,D}", r"\Big]",
            r"\in", r"\{0, 1\}^D",
            font_size=34
        )
        vec_tex.set_color_by_tex(r"\vec{X}_i", BLUE_B)
        vec_tex.set_color_by_tex(r"\{0, 1\}^D", TEAL_B)
        vec_tex.next_to(sec1_label, DOWN, buff=0.3)

        bit_meaning = MathTex(
            r"x_{i,j} = \begin{cases} 1 & \text{if feature } j \text{ is selected} \\ 0 & \text{if feature } j \text{ is unselected} \end{cases}",
            font_size=24,
            color=LIGHT_GRAY
        )
        bit_meaning.next_to(vec_tex, DOWN, buff=0.35)

        self.play(Write(sec1_label), run_time=0.7)
        self.play(Write(vec_tex), run_time=1.0)
        self.play(FadeIn(bit_meaning, shift=UP * 0.2), run_time=0.9)
        self.wait(1.5)

        # Clean section 1 to make room for Fitness Function
        self.play(
            FadeOut(sec1_label),
            FadeOut(bit_meaning),
            vec_tex.animate.scale(0.8).to_corner(UL, buff=0.8),
            run_time=0.8
        )

        # ---------------------------------------------------------------------
        # 3. Fitness Function Morphing Sequence
        # ---------------------------------------------------------------------
        sec2_label = Text("2. Multi-Objective Fitness Evaluation", font_size=22, color=YELLOW_C)
        sec2_label.next_to(header, DOWN, buff=0.35)
        self.play(Write(sec2_label), run_time=0.6)

        # Step A: Raw Intuitive Concept
        f_concept = MathTex(
            r"f(\vec{X})", r"=", r"\text{Maximize}\Big(", r"\text{Accuracy}", r"\Big)",
            r"-", r"\text{Minimize}\Big(", r"\text{Selected Features}", r"\Big)",
            font_size=32
        )
        f_concept.set_color_by_tex(r"Accuracy", GREEN_C)
        f_concept.set_color_by_tex(r"Selected Features", RED_C)
        f_concept.move_to(ORIGIN + UP * 0.3)

        self.play(Write(f_concept), run_time=1.2)
        self.wait(1.2)

        # Step B: Formal Weighted Multi-Objective Form
        f_formal = MathTex(
            r"f(\vec{X})", r"=", r"w_1 \cdot \text{Acc}_{k\text{-NN}}(\vec{X})",
            r"+", r"w_2 \cdot \left(1 - \frac{|\vec{X}|}{D}\right)",
            font_size=34
        )
        f_formal.set_color_by_tex(r"\text{Acc}_{k\text{-NN}}", GREEN_C)
        f_formal.set_color_by_tex(r"\frac{|\vec{X}|}{D}", PURPLE_B)
        f_formal.move_to(ORIGIN + UP * 0.3)

        self.play(
            FadeTransform(f_concept, f_formal),
            run_time=1.2
        )
        self.wait(1.0)

        # Step C: Exact Baseline Paper Equation (with w2 = 0.001)
        f_baseline = MathTex(
            r"\max_{\vec{X}} f(\vec{X})", r"=",
            r"\overline{\text{Accuracy}}(\vec{X})",
            r"+",
            r"0.001 \cdot \left(1 - \frac{|S|}{D}\right)",
            font_size=36
        )
        f_baseline.set_color_by_tex(r"\max_{\vec{X}} f(\vec{X})", GOLD_B)
        f_baseline.set_color_by_tex(r"\overline{\text{Accuracy}}", GREEN_B)
        f_baseline.set_color_by_tex(r"0.001", TEAL_B)
        f_baseline.set_color_by_tex(r"\frac{|S|}{D}", PINK)
        f_baseline.move_to(ORIGIN + UP * 0.3)

        box_fitness = SurroundingRectangle(f_baseline, color=YELLOW_B, buff=0.2, corner_radius=0.1)
        
        desc_acc = MathTex(r"\text{10-Fold CV Classification Accuracy}", font_size=20, color=GREEN_C)
        desc_red = MathTex(r"\text{Feature Reduction Ratio Penalty } (|S| = \text{subset size}, D = \text{total features})", font_size=20, color=PINK)
        
        annotations = VGroup(desc_acc, desc_red).arrange(DOWN, buff=0.15).next_to(box_fitness, DOWN, buff=0.3)

        self.play(
            FadeTransform(f_formal, f_baseline),
            Create(box_fitness),
            run_time=1.2
        )
        self.play(FadeIn(annotations, shift=UP * 0.15), run_time=0.8)
        self.wait(2.0)

        # Clean section 2
        self.play(
            FadeOut(sec2_label),
            FadeOut(box_fitness),
            FadeOut(annotations),
            FadeOut(f_baseline),
            FadeOut(vec_tex),
            run_time=0.8
        )

        # ---------------------------------------------------------------------
        # 4. Position Update Equations (Lévy Flight & Abandonment)
        # ---------------------------------------------------------------------
        sec3_label = Text("3. Search Dynamics: Lévy Flight & Nest Abandonment", font_size=22, color=YELLOW_C)
        sec3_label.next_to(header, DOWN, buff=0.35)
        self.play(Write(sec3_label), run_time=0.6)

        # Continuous Lévy Update
        eq_levy = MathTex(
            r"\vec{x}_i^{(t+1)}", r"=", r"\vec{x}_i^{(t)}", r"+", r"\alpha \cdot \text{step} \odot \text{L\'evy}(\lambda)",
            font_size=32
        )
        eq_levy.set_color_by_tex(r"\vec{x}_i^{(t+1)}", BLUE_C)
        eq_levy.set_color_by_tex(r"\text{L\'evy}(\lambda)", RED_B)
        eq_levy.move_to(ORIGIN + UP * 1.0)

        levy_scaling = MathTex(
            r"\text{where } \text{L\'evy}(\lambda) \sim t^{-\lambda}, \quad 1 < \lambda \le 3",
            font_size=24,
            color=GRAY_B
        ).next_to(eq_levy, DOWN, buff=0.2)

        # Nest Abandonment Update
        eq_abandon = MathTex(
            r"\vec{x}_i^{(t+1)}", r"=", r"\vec{x}_i^{(t)}", r"+", r"r \cdot \delta \left( \vec{x}_{r_1}^{(t)} - \vec{x}_{r_2}^{(t)} \right)",
            r"\quad \text{with probability } p_a",
            font_size=30
        )
        eq_abandon.set_color_by_tex(r"p_a", ORANGE)
        eq_abandon.set_color_by_tex(r"\vec{x}_{r_1}^{(t)} - \vec{x}_{r_2}^{(t)}", TEAL_C)
        eq_abandon.next_to(levy_scaling, DOWN, buff=0.55)

        self.play(Write(eq_levy), FadeIn(levy_scaling, shift=UP * 0.1), run_time=1.2)
        self.wait(1.0)
        self.play(Write(eq_abandon), run_time=1.2)
        self.wait(1.5)

        # Transition arrow to discretization
        trans_box = RoundedRectangle(
            height=1.2, width=8.5, corner_radius=0.15,
            color=PURPLE_B, fill_color="#181a27", fill_opacity=0.8
        ).to_edge(DOWN, buff=0.6)

        trans_text = Text(
            "Continuous position values are mapped back to binary {0, 1} bits\nusing the V-shaped transfer function: V2 = |tan(x)|",
            font_size=20,
            color=WHITE,
            line_spacing=1.2
        ).move_to(trans_box.get_center())

        self.play(
            Create(trans_box),
            Write(trans_text),
            run_time=1.2
        )
        self.wait(2.5)

        # Fade out all
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=1.0)
