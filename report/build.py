"""Build the written submission from proofs and saved measured results."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from task2.parameter_count import calculate
from task5.evaluate import evaluate
from task5.model import Acceleration


def build():
    trace = json.loads(Path("results/task6_trace.json").read_text())
    task6_success = trace.get("status") == "success" and trace.get("valid_finish_condition_met", False)
    font = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("DejaVu", str(font / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuBold", str(font / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFontFamily(
        "DejaVu",
        normal="DejaVu",
        bold="DejaVuBold",
        italic="DejaVu",
        boldItalic="DejaVuBold",
    )
    styles = getSampleStyleSheet()
    for name in styles.byName:
        styles[name].fontName = "DejaVu"
        styles[name].leading = 12.5
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].spaceAfter = 6
    styles["Heading1"].leading = 23
    styles["Heading2"].leading = 16
    styles.add(
        ParagraphStyle(
            "Equation",
            fontName="DejaVu",
            fontSize=10.5,
            leading=15,
            leftIndent=14,
            spaceAfter=9,
        )
    )
    story = []

    def p(text):
        story.append(Paragraph(text, styles["BodyText"]))

    def eq(text):
        story.append(Paragraph(text, styles["Equation"]))

    def title(text):
        story.append(Paragraph(text, styles["Heading1"]))
        story.append(Spacer(1, 12))

    def sub(text):
        story.append(Paragraph(text, styles["Heading2"]))

    def page():
        story.append(PageBreak())

    def table(rows, widths=None):
        formatted = [
            [
                Paragraph(
                    escape(str(cell)),
                    ParagraphStyle("cell", fontName="DejaVu", fontSize=8, leading=11),
                )
                for cell in row
            ]
            for row in rows
        ]
        t = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a7b4c1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 12))

    title("Homework 1")
    p("AI, Neural Networks, Large Language Models, and Agentic AI Systems")
    p("<b>Student: Ara Davtyan</b><br/>Prepared: 13 October 2026")
    p(
        "Tasks 1–4 contain direct constructions and proofs. Task 5 contains real Silverbox experiments. "
        + ("Task 6 completed a real model-driven run with verified execution evidence. " if task6_success else "Task 6 infrastructure is implemented and tested; its real workflow has not completed successfully. ")
        + "The Task 6 section records the actual latest run status. GitHub publication remains pending."
    )
    sub("Task 1. ReLU networks")
    p(
        "Write R(z)=max(0,z). Hidden neurons use R; every output below uses the identity activation. No skip connections are used."
    )
    p(
        "<b>(a) Absolute value.</b> The two hidden neurons have weights 1 and −1, and biases 0 and 0. The output weights are (1,1), and the output bias is 0:"
    )
    eq("h₁=R(x),  h₂=R(−x),  f(x)=h₁+h₂=|x|.")
    p(
        "For x≥0 the hidden values are (x,0); for x≤0 they are (0,−x). Their sum is |x| on the entire real line, including zero. Two hidden units are minimal: with one hidden ReLU, A R(wx+b)+c is constant on one unbounded half-line (or everywhere if w=0), whereas |x| is nonconstant on both tails."
    )
    table(
        [
            ["x", "Hidden pre-activations", "Hidden activations", "Output"],
            [-2, "(−2, 2)", "(0, 2)", 2],
            [-0.5, "(−0.5, 0.5)", "(0, 0.5)", 0.5],
            [0, "(0, 0)", "(0, 0)", 0],
            [0.5, "(0.5, −0.5)", "(0.5, 0)", 0.5],
            [3, "(3, −3)", "(3, 0)", 3],
        ],
        [50, 160, 160, 90],
    )
    p(
        "<b>(b) XOR.</b> Hidden weights are (1,−1) and (−1,1), both biases 0. Output weights are (1,1), bias 0, with identity activation:"
    )
    eq("h₁=R(x₁−x₂), h₂=R(x₂−x₁), f=h₁+h₂=|x₁−x₂|.")
    table(
        [
            ["Input", "(h₁,h₂)", "Output"],
            ["(0,0)", "(0,0)", 0],
            ["(0,1)", "(0,1)", 1],
            ["(1,0)", "(1,0)", 1],
            ["(1,1)", "(0,0)", 0],
        ],
        [150, 160, 150],
    )
    p(
        "Two hidden units are minimal: thresholding a one-ReLU affine-output network at 1/2 gives a half-space, but XOR labels opposite corners alike, and the two diagonal pairs have the same midpoint, so no affine separator exists. On Boolean inputs, absolute difference is exactly XOR."
    )

    page()
    title("Task 2. Parameter counting")
    p(
        "For each spatial layer, H<sub>out</sub>=⌊(H+2p−k)/s⌋+1 and likewise for width. A convolution has k²C<sub>in</sub>C<sub>out</sub> weights and C<sub>out</sub> biases. A dense layer has N<sub>in</sub>N<sub>out</sub> weights and N<sub>out</sub> biases. ReLU and max-pooling each contribute zero parameters."
    )
    rows = calculate()
    formulas = [
        "(224+6−7)//2+1=112",
        "(112−3)//2+1=55",
        "(55+4−5)//1+1=55",
        "(55−3)//2+1=27",
        "(27+2−3)//1+1=27",
        "(27+2−3)//1+1=27",
        "(27+2−3)//1+1=27",
        "(27−3)//2+1=13",
    ]
    table(
        [["Layer", "Spatial calculation", "Output shape"]]
        + [
            [
                r["layer"],
                formulas[i] if i < 8 else "Dense",
                " × ".join(map(str, r["shape"])),
            ]
            for i, r in enumerate(rows)
        ],
        [55, 245, 160],
    )
    weight_formulas = [
        "7²×3×64",
        "0",
        "5²×64×192",
        "0",
        "3²×192×384",
        "3²×384×256",
        "3²×256×256",
        "0",
        "43,264×512",
        "512×256",
        "256×10",
    ]
    table(
        [["Layer", "Weight formula", "Weights", "Biases", "Total"]]
        + [
            [
                r["layer"],
                weight_formulas[i],
                f"{r['weights']:,}",
                f"{r['biases']:,}",
                f"{r['parameters']:,}",
            ]
            for i, r in enumerate(rows)
        ],
        [45, 130, 105, 65, 115],
    )
    eq(
        "Flattened dimension = 13×13×256 = 43,264.<br/>Total = 24,741,450 trainable parameters."
    )
    p(
        "The independent Python formula calculation was compared against executed PyTorch Conv2d, MaxPool2d, Flatten and Linear layers, checking every intermediate shape and every layer count. ReLUs do not change shapes or counts. All comparisons agreed exactly; this is the assigned architecture, not the published AlexNet count."
    )

    page()
    title("Task 3. Softmax is a mass function")
    p(
        "Let K≥1 be finite and z∈ℝ<super>K</super> have finite entries. Set Z=Σ<sub>j=1</sub><super>K</super> exp(z<sub>j</sub>). Each exponential is strictly positive and finite; hence 0&lt;Z&lt;∞."
    )
    eq("s<sub>i</sub>(z)=exp(z<sub>i</sub>)/Z &gt; 0,  i=1,…,K.")
    eq(
        "Σ<sub>i=1</sub><super>K</super> s<sub>i</sub>(z)<br/>= (Σ<sub>i=1</sub><super>K</super> exp(z<sub>i</sub>))/Z = Z/Z = 1."
    )
    p(
        "Therefore, for each fixed score vector z, s(z) defines a probability mass function on the finite class set {1,…,K}. The probability of a set of classes A is Σ<sub>i∈A</sub>s<sub>i</sub>(z)."
    )
    sub("Why it is not a continuous probability density")
    p(
        "The normalization above sums over discrete class indices. It does not integrate over a continuous outcome. Although z varies continuously, z is a conditioning score vector, not the outcome whose probabilities are being assigned. Softmax produces one mass for each class; it introduces no real-valued outcome variable t and no density f(t) satisfying ∫ f(t)dt=1."
    )
    p(
        "A mass function can be called a density relative to counting measure in measure-theoretic terminology. Here “density” means the continuous-outcome density defined in the question; the distinction is between summation over classes and integration over continuous outcomes."
    )
    p(
        "The numerical check uses max-subtraction for stability and verifies positivity and normalization on finite moderate scores. Strict positivity is an exact real-arithmetic property; floating-point exponentials can underflow to zero for extreme score differences."
    )

    page()
    title("Task 4. Attention and permutations")
    p(
        "Rows of X are token vectors. Q=XW<sub>Q</sub>, K=XW<sub>K</sub>, V=XW<sub>V</sub>. Let P<sub>i,π(i)</sub>=1. Attention is unmasked; softmax acts row-wise."
    )
    sub("(a) Scores follow token identities")
    eq(
        "(QKᵀ)<sub>ij</sub> = Σ<sub>r</sub> Q<sub>ir</sub>K<sub>jr</sub> = q<sub>i</sub>ᵀk<sub>j</sub>."
    )
    p(
        "Since Q′=(PX)W<sub>Q</sub>=PQ and K′=PK, their i-th and j-th rows are q<sub>π(i)</sub> and k<sub>π(j)</sub>. Thus:"
    )
    eq("(q′<sub>i</sub>)ᵀk′<sub>j</sub> = q<sub>π(i)</sub>ᵀk<sub>π(j)</sub>.")
    p(
        "When a token moves from seat 1 to seat 3, every dot product of that token with the other tokens retains its value; only the row and column seats occupied by those values change."
    )
    p(
        "For u=(1,0), v=(0,1), W<sub>Q</sub>=W<sub>K</sub>=I, the score table in seat order (u,v) is [[1,0],[0,1]]. After swapping to (v,u), it is also [[1,0],[0,1]]. The u·u entry moves from (1,1) to (2,2), v·v moves conversely, and u·v and v·u exchange (1,2) and (2,1). Equality of the displayed matrices is due to this symmetric example."
    )
    sub("(b) Equivariance of the whole layer")
    eq("Q′=PQ, K′=PK, V′=PV.<br/>S′=Q′K′ᵀ/√d<sub>k</sub>=P S Pᵀ.")
    p(
        "For A=softmax(S), entry A<sub>ij</sub>=exp(S<sub>ij</sub>)/Σ<sub>l</sub>exp(S<sub>il</sub>). Reindexing l by a permutation leaves the denominator unchanged, so softmax(PSPᵀ)=PAPᵀ. Consequently:"
    )
    eq("Att(PX)=(PAPᵀ)(PV)=PA(PᵀP)V=PAV=P Att(X).")
    p(
        "The output row associated with each token moves with that token. With W<sub>V</sub>=I and the scaling omitted only for this numerical example, let a=e/(e+1)=0.73105858 and b=1/(e+1)=0.26894142:"
    )
    eq("Att(X)=[[a,b],[b,a]],<br/>Att(PX)=[[b,a],[a,b]]=P Att(X).")
    sub("(c) Position must affect the dot-product inputs")
    eq("PX+E has rows (x₂+e₁, x₁+e₂).<br/>P(X+E) has rows (x₂+e₂, x₁+e₁).")
    p(
        "Their difference is E−PE, with rows (e₁−e₂,e₂−e₁), which is nonzero if e₁≠e₂. Thus PX+E≠P(X+E). The seat-specific score is (x<sub>i</sub>+e<sub>i</sub>)ᵀ(x<sub>j</sub>+e<sub>j</sub>), so seat information is already present before mixing values. Shared row-wise layers commute with P; composing them with unmasked attention preserves F(PX)=PF(X) absent positional information. The input inequality does not imply every choice of weights produces distinct outputs."
    )

    page()
    title("Task 5. Silverbox neural acceleration")
    dataset = json.loads(Path("results/dataset.json").read_text())
    experiments = json.loads(Path("results/experiments.json").read_text())
    best = json.loads(Path("results/best.json").read_text())
    current = evaluate(Path("results") / (best["name"] + ".pt"), plot=True)
    for key, value in current.items():
        if value != best[key]:
            raise ValueError(
                f"Best result {key} disagrees with current checkpoint evaluation"
            )
    architecture = "→".join(map(str, [3, *best["widths"], 1]))
    parameter_count = sum(p.numel() for p in Acceleration(best["widths"]).parameters())
    p(
        "The official Silverbox benchmark MAT file contains V1 and V2, each with shape (1,131072), plus overload indicators. V1 is the applied voltage u and V2 the measured output y. The exact slice [42650:46746] contains 4096 samples. Each channel is centered by its own mean over that whole excerpt, as the question requires. Training uses samples 0–3071 and testing 3072–4095 of the excerpt."
    )
    p(
        "Data source: https://www.nonlinearbenchmark.org/benchmarks/silverbox. The official nonlinear-benchmarks loader downloads the linked archive. The downloaded file’s SHA-256 is recorded for reproducibility, not asserted against an unavailable publisher checksum:"
    )
    p(dataset["sha256"][:32] + "<br/>" + dataset["sha256"][32:])
    eq(
        "f<sub>s</sub>=610.35 Hz, Δt=1/f<sub>s</sub>≈0.00163840419 s.<br/>ÿ+aẏ+by+cy³=gu(t).<br/>ẏ=v,  v̇=α<sub>θ</sub>(y,v,u)."
    )
    p(
        "The physical terms represent damping, linear stiffness, cubic stiffness and input gain. They motivate an acceleration function; the network weights are not the four physical coefficients. The implemented model learns acceleration directly, rather than fitting an unrelated static mapping or using a physics residual as a replacement for integration."
    )
    p(
        f"For the kept model, normalized inputs are (y/0.1, v/45, u/0.1), in volts, volts/second and volts respectively. The fully connected map is {architecture} with tanh after each hidden affine layer and a linear output multiplied by 20,000 V/s². Its trainable count is {parameter_count:,}. Initialization is PyTorch Linear’s seeded default. The fixed scale buffers are saved with the weights and used identically during inference."
    )
    sub("Differentiable integration and sample alignment")
    p(
        "Use classical RK4 on s=(y,v), with F(s,u)=(v,α<sub>θ</sub>(s,u)). At a step from sample i to i+1, u at the first and fourth stages is u<sub>i</sub> and u<sub>i+1</sub>; both middle stages use (u<sub>i</sub>+u<sub>i+1</sub>)/2, i.e. piecewise linear input interpolation."
    )
    eq(
        "k₁=F(s<sub>i</sub>,u<sub>i</sub>),<br/>k₂=F(s<sub>i</sub>+Δt k₁/2,(u<sub>i</sub>+u<sub>i+1</sub>)/2),<br/>k₃=F(s<sub>i</sub>+Δt k₂/2,(u<sub>i</sub>+u<sub>i+1</sub>)/2),<br/>k₄=F(s<sub>i</sub>+Δt k₃,u<sub>i+1</sub>),<br/>s<sub>i+1</sub>=s<sub>i</sub>+Δt(k₁+2k₂+2k₃+k₄)/6."
    )
    p(
        "The first predicted output is the initial position; N samples require N−1 integration steps. Final evaluation initializes s₀=(y₀,(y₁−y₀)/Δt), and then makes one 4096-point rollout. The state is carried across index 3072 without a reset. The prediction function receives only the input record and the first two measured positions. Measured test outputs are used only afterwards to calculate errors and plots. Whole-excerpt centering is the assignment’s prescribed preprocessing."
    )

    page()
    title("Task 5. Training and measured results")
    p(
        "Only training observations initialize training windows. Training velocity and acceleration estimates come from a Savitzky–Golay filter (9 samples, degree 5, first/second derivatives, Δt). These estimates warm up the acceleration network for 1,200 optimizer steps, excluding five points at each edge. Then optimize multistep rollout loss mean[((ŷ−y)/0.1)²], with 32 independently sampled windows per batch. Training windows avoid the filtered edges; none includes test samples. AdamW uses PyTorch’s default weight decay of 0.01; Adam uses zero weight decay."
    )
    p(
        "All runs use seed 42, float32 and one CPU thread. Multistep updates clip the gradient norm at 10 and check finite loss/gradients. The audited implementation also checks finite warm-up loss/gradients without clipping their norm. The window horizon includes the initial sample. Both architectures were compared under multiple rates, horizons, optimizers and update counts. Only completed runs enter the table."
    )
    table(
        [
            [
                "Widths",
                "Optimizer / rate",
                "Updates / horizon",
                "Train RMSE (V)",
                "Test RMSE (V)",
                "Time (s)",
            ]
        ]
        + [
            [
                str(r["widths"]),
                r["optimizer"] + " / " + str(r["learning_rate"]),
                str(r["training_steps"]) + " / " + str(r["rollout_length"]),
                f"{r['train_rmse']:.6f}",
                f"{r['test_rmse']:.6f}",
                f"{r['elapsed_time']:.1f}",
            ]
            for r in experiments
            if r["result_status"] == "completed"
        ],
        [65, 112, 80, 77, 77, 49],
    )
    p(
        f"Kept run: <b>{best['name']}</b>. Optimizer {best['optimizer']}, rate {best['learning_rate']}, {best['training_steps']} multistep updates after {best['pretrain_steps']} warm-up updates, horizon {best['rollout_length']}, batch size {best['batch_size']}. Selection uses measured test RMSE as requested; this score is not an untouched independent estimate. A fresh same-seed training run reproduced every weight tensor and metric exactly. An independent RK4 replay also reproduced all six saved prediction arrays and metrics exactly."
    )
    eq(
        f"Full-free-run train RMSE = {best['train_rmse']:.9f} V.<br/>Full-free-run test RMSE = {best['test_rmse']:.9f} V.<br/>Zero-prediction test baseline = {best['zero_test_rmse']:.9f} V."
    )
    p(
        "The baseline agrees with the approximately 0.06 V stated in the problem. RMSE is sqrt(mean((prediction−measurement)²)), separately over the first 3072 and final 1024 predictions of the same run. The best network improves substantially on zero, with a larger error on testing than training. No additional coefficient recovery or unseen-record performance is claimed."
    )
    story.append(
        Image(str(Path("results") / (best["name"] + ".png")), width=480, height=144)
    )
    p(
        "Figure: measured and predicted centered output for the complete test segment. Time is relative to the excerpt start; voltage units are V. This prediction is the uninterrupted free run from sample zero. Saved JSON records contain the measured metrics; the final checkpoint and evaluation command reproduce the prediction and plot."
    )

    page()
    title("Task 6. Agentic system and validation status")
    p(
        "The implementation uses LangGraph StateGraph with exactly four role nodes: orchestrator, scientist, general and executor. START enters the orchestrator; conditional edges choose a specialist or END; specialist edges return to the orchestrator. State contains chronological {role,text} dialogue, decisions, execution evidence and status. Every role receives prior dialogue, including the latest scientist plan."
    )
    table(
        [
            ["Role", "Responsibility / access"],
            [
                "orchestrator",
                "Real LLM structured decision: next_role, instruction, reason. Routes only; no coding tool. At most eight decisions.",
            ],
            [
                "scientist",
                "Duffing equation, neural architecture, state formulation, training, RK4, continuous free-run evaluation and numbered implementation plan. No coding tool.",
            ],
            [
                "general",
                "Supporting conceptual questions; no fitting plan or coding tool.",
            ],
            [
                "executor",
                "Generates source implementing the latest scientist plan. Only role with the tool. Writes and runs Python; failures and repairs remain in dialogue.",
            ],
        ],
        [95, 385],
    )
    p(
        "The decision schema rejects invalid roles and missing/empty fields. A generated textual RMSE is insufficient: the executor must save a fresh compatible neural checkpoint, and a separately executed trusted script must print a real full-free-run metric. Successful verification is stored in graph state; only then may finish succeed. A premature finish or exhausted decision limit gives an unsuccessful status."
    )
    p(
        "The tool creates a run-specific directory, writes solution.py, and launches Python with subprocess without a shell. It captures real exit status, stdout and stderr, kills the process group on timeout, and returns actual tracebacks. The whole call, including the independent evaluator, has a 300-second budget. Credentials are stripped from the child environment. Path auditing restricts reads and writes, blocks subprocesses and sockets, and Linux limits constrain CPU, address space and output file size. Python auditing is a practical safeguard, not secure isolation against hostile native code; a container or VM is required for that use."
    )
    sub("Actual Task 5 demonstration attempt")
    p(
        f"Saved trace status: <b>{escape(trace['status'])}</b>. {escape(trace.get('error', ''))}."
    )
    roles = [role for event in trace.get("events", []) for role in event]
    p(f"Real completed model calls: {trace.get('real_llm_calls_completed', 0)}. Orchestrator decisions: {trace.get('orchestrator_decisions', 0)}. Role transitions: {escape(' → '.join(roles) or 'none completed')}.")
    if task6_success:
        p(f"The executor actually ran a script and the trusted continuous free-run evaluator printed test RMSE {trace['test_rmse']:.9f} V. The orchestrator finished only after verified execution. Complete decisions, plans, source and actual process output are retained in results/task6_trace.json.")
        for event in trace.get("events", []):
            for role, update in event.items():
                if role in {"scientist", "general"} and update.get("dialogue"):
                    p(f"<b>{escape(role)}:</b> " + escape(update["dialogue"][-1]["text"][:1600]).replace('\n', '<br/>'))
    else:
        p("The actual Task 5 request was supplied to task6.run, but no successful end-to-end result is claimed. The saved error above identifies the actual blocker; configured credentials alone do not guarantee API credit or execution success. The Task 5 metrics elsewhere in this report come from direct experiments, not from the agent workflow.")
    sub("Infrastructure tests are separate evidence")
    p(
        "Mock role outputs test all graph routes, chronological role attribution, the eight-decision limit, rejection of finish without execution, and rejection of execution without a scientist plan. Real small subprocess tests verify source writing, stdout/stderr, errors, timeout and out-of-directory write rejection. Another test creates an untrained zero-weight checkpoint and actually runs the trusted evaluator. Its metric is a unit-test artifact, not a fitted Silverbox agent result. The complete real graph demonstration remains required."
    )

    page()
    title("Reproduction and remaining submission requirements")
    p(
        "Use Python 3.13.9 on Linux; main dependency versions are pinned in requirements.txt. CPU training is sufficient. From the repository root, create/activate .venv and install requirements.txt with the PyTorch CPU index. The README gives complete installation details."
    )
    for command in [
        "python -m task5.data",
        "python -m task1.verify_networks",
        "python -m task2.parameter_count",
        "python -m task3.verify_softmax",
        "python -m task4.verify_attention",
        "python -m task5.experiments",
        f"python -m task5.evaluate results/{best['name']}.pt",
        "python -m pytest -q",
        "python -m task6.run",
        "python -m report.build",
    ]:
        p('<font size="9">' + escape(command) + "</font>")
    p(
        "The experiment sweep reuses a completed run only if its checkpoint exists, its configuration matches, and reevaluated metrics match the JSON record. Invalid or missing artifacts trigger retraining, with old records retained in history. The selected checkpoint is included in Git; data and other checkpoints are excluded. Rerun acquisition and the experiment sweep to reproduce all comparisons. The final technical audit is recorded in report/verification.md."
    )
    p(
        "<b>GitHub repository:</b> publication blocked. No usable repository metadata, authenticated GitHub account or destination URL was available. No push was attempted and no URL is invented. Tasks 5 and 6 must be published and their verified repository URL added to this PDF before final submission."
    )
    p(
        "<b>Task 6:</b> a valid locally configured API key, available API credits and network access are required. If the saved run is unsuccessful, resolve its recorded error and rerun task6.run. Unit tests do not replace the real workflow requirement."
    )
    p(
        "<b>AI assistance disclosure:</b> AI assistance generated and checked the mathematical solutions, implementation, experiments, tests and report. This document makes no claim that the student independently performed those actions. The student should review the work and follow the lecturer’s applicable disclosure rules. No additional lecturer AI-use policy was present in the provided seven-page assignment."
    )
    p(
        "The original assignment requests one solution PDF, code on GitHub for Tasks 5 and 6, and email submission. This package provides the PDF locally but does not claim publication or email delivery."
    )
    sub("Sources")
    p(
        "Assignment: homework-1.pdf, seven pages, supplied by the user.<br/>Silverbox: nonlinearbenchmark.org/benchmarks/silverbox.<br/>Wigren and Schoukens (2013), Three free data sets for development and benchmarking in nonlinear system identification, ECC, pp. 2933–2938.<br/>LangGraph Graph API: docs.langchain.com/oss/python/langgraph/graph-api.<br/>LangChain structured output: docs.langchain.com/oss/python/langchain/structured-output."
    )

    def footer(canvas, doc):
        canvas.setFont("DejaVu", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(48, 26, "Ara Davtyan · Homework 1")
        canvas.drawRightString(547, 26, str(doc.page))

    SimpleDocTemplate(
        "report/homework1.pdf",
        pagesize=(595, 842),
        rightMargin=48,
        leftMargin=48,
        topMargin=40,
        bottomMargin=45,
        title="Homework 1 — Ara Davtyan",
        author="Ara Davtyan",
    ).build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
