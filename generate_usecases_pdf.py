"""Generate a PDF of Sleeping LLM use cases and killer advantages."""

from fpdf import FPDF

FONT = "Arial"
FONTS_DIR = "/System/Library/Fonts/Supplemental"


class UseCasesPDF(FPDF):
    def setup_fonts(self):
        self.add_font(FONT, "", f"{FONTS_DIR}/Arial Unicode.ttf", uni=True)
        self.add_font(FONT, "B", f"{FONTS_DIR}/Arial Bold.ttf", uni=True)
        self.add_font(FONT, "I", f"{FONTS_DIR}/Arial Italic.ttf", uni=True)
        self.add_font(FONT, "BI", f"{FONTS_DIR}/Arial Bold Italic.ttf", uni=True)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font(FONT, "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, "Sleeping LLM \u2014 Use Cases & Market Advantages", align="R")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT, "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font(FONT, "B", 16)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, title)
        self.ln(12)

    def subsection_title(self, title):
        self.set_font(FONT, "B", 13)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title)
        self.ln(10)

    def body_text(self, text):
        self.set_font(FONT, "", 10.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font(FONT, "B", 10.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def italic_text(self, text):
        self.set_font(FONT, "I", 10.5)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font(FONT, "", 10.5)
        self.set_text_color(40, 40, 40)
        self.cell(8, 5.5, "\u2022")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def highlight_box(self, text):
        self.set_fill_color(240, 245, 250)
        self.set_font(FONT, "B", 11)
        self.set_text_color(30, 60, 120)
        self.set_x(15)
        self.multi_cell(self.w - 30, 8, text, fill=True, align="C")
        self.ln(4)

    def table_row(self, cols, widths, bold=False):
        style = "B" if bold else ""
        self.set_font(FONT, style, 9.5)
        self.set_text_color(40, 40, 40)
        h = 7
        for i, col in enumerate(cols):
            if bold:
                self.set_fill_color(230, 235, 245)
                self.cell(widths[i], h, col, border=1, fill=True)
            else:
                self.cell(widths[i], h, col, border=1)
        self.ln(h)

    def who_pays(self, text):
        self.set_font(FONT, "B", 10)
        self.set_text_color(30, 100, 60)
        self.cell(0, 6, f"Who pays: {text}")
        self.ln(8)


def build_pdf():
    pdf = UseCasesPDF()
    pdf.setup_fonts()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # --- Title page ---
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font(FONT, "B", 28)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 15, "Sleeping LLM", align="C")
    pdf.ln(18)
    pdf.set_font(FONT, "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Use Cases & Killer Advantages", align="C")
    pdf.ln(12)
    pdf.cell(0, 10, "for Early Adopters", align="C")
    pdf.ln(30)
    pdf.set_font(FONT, "I", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Weight-based persistent memory for LLMs", align="C")
    pdf.ln(8)
    pdf.cell(0, 8, "Beyond RAG: knowledge in the weights, not the database", align="C")
    pdf.ln(40)
    pdf.set_font(FONT, "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, "Vladimir Baranov", align="C")
    pdf.ln(6)
    pdf.cell(0, 6, "February 2026", align="C")

    # --- Core Advantage ---
    pdf.add_page()
    pdf.section_title("The Core Advantage")

    pdf.highlight_box("RAG answers: \"Let me look that up for you.\"    This answers: \"I already know that.\"")

    pdf.body_text(
        "The difference is not cosmetic. It changes what is possible. "
        "RAG retrieves text snippets from a database and pastes them into the context window. "
        "A sleeping LLM consolidates knowledge directly into model weights through LoRA "
        "fine-tuning during offline \"sleep\" cycles. After sleep, the context window is empty "
        "and the model genuinely knows things it learned from conversation."
    )
    pdf.ln(2)
    pdf.body_text(
        "This means: no retrieval latency, no context window consumed by retrieved documents, "
        "knowledge that generalizes (the model can reason with learned facts, combine them, "
        "make novel inferences), personality and style that evolve alongside factual knowledge, "
        "and full privacy with no external database to breach."
    )

    # --- Use Cases ---
    pdf.ln(4)
    pdf.section_title("Killer Use Cases")

    # 1
    pdf.subsection_title("1. Personalized Medical / Therapy Companion")
    pdf.body_text(
        "A therapist who remembers your history does not check a file before each session "
        "-- they know you. The patterns, triggers, and progress are internalized. A RAG system "
        "retrieves \"patient mentioned anxiety on Jan 3.\" A sleeping LLM develops an intuitive "
        "model of the person."
    )
    pdf.bullet("Remembers medication changes, symptoms, emotional patterns across months")
    pdf.bullet("Develops a personalized conversational style adapted to the user")
    pdf.bullet("Knowledge generalizes: connects a sleep complaint in March to a job change in January without explicit retrieval")
    pdf.bullet("Fully local -- mental health data never leaves the device")
    pdf.who_pays("Digital health companies, elder care, addiction recovery programs")

    # 2
    pdf.subsection_title("2. Personal Engineering Copilot That Learns YOUR Codebase")
    pdf.body_text(
        "Current copilots treat every session as day one. RAG can retrieve snippets, but it "
        "does not understand your architecture. A sleeping LLM that has been through 50 sleep "
        "cycles on your codebase conversations has internalized your naming conventions, "
        "architectural patterns, common bugs, and preferences."
    )
    pdf.bullet("\"Use our Redis cache pattern\" -- it knows what that means without retrieval")
    pdf.bullet("Learns your code review preferences, your team's style guide, your deployment quirks")
    pdf.bullet("Does not consume context window with retrieved docs -- the knowledge IS the model")
    pdf.bullet("Gets better every week, not just every time the vendor ships an update")
    pdf.who_pays("Dev teams, enterprise software companies, solo developers")

    # 3
    pdf.subsection_title("3. Domain Expert That Never Forgets a Client")
    pdf.body_text(
        "A financial advisor, lawyer, or consultant who remembers every client interaction, "
        "every decision made, every preference stated -- without checking notes. The model "
        "develops genuine expertise in YOUR situation."
    )
    pdf.bullet("\"Given what we discussed about your risk tolerance and the tax implications from last quarter...\" -- no retrieval, just knowledge")
    pdf.bullet("Learns industry jargon, regulatory nuances, client-specific context over time")
    pdf.bullet("Can reason across multiple clients' patterns (anonymized) to spot opportunities")
    pdf.bullet("Scales expertise: one model per client, each accumulating domain knowledge")
    pdf.who_pays("Wealth management firms, law firms, consulting companies")

    # 4
    pdf.add_page()
    pdf.subsection_title("4. Companion AI with Genuine Personality Development")
    pdf.body_text(
        "This is the one RAG fundamentally cannot do. RAG retrieves facts -- it does not "
        "change who the model IS. A sleeping LLM's personality, humor, interests, and "
        "conversational style evolve through weight updates. The model does not just remember "
        "that you like dry humor -- it becomes funnier in the way you appreciate."
    )
    pdf.bullet("Relationship deepens over time -- not simulated via prompt engineering")
    pdf.bullet("Shared references and inside jokes emerge naturally")
    pdf.bullet("The model's personality genuinely adapts, not just its fact retrieval")
    pdf.bullet("Users feel the difference immediately -- \"it actually knows me\"")
    pdf.who_pays("Consumer AI companies, elder care, education, children's AI tutors")

    # 5
    pdf.subsection_title("5. On-Device Intelligence for Privacy-Critical Applications")
    pdf.body_text(
        "The model runs locally, learns locally, and the knowledge lives in weights on the "
        "device. No database to breach, no API calls to intercept, no server logs."
    )
    pdf.bullet("Military/intelligence analysts: learns patterns across classified briefings")
    pdf.bullet("Journalists: builds source knowledge that cannot be subpoenaed from a cloud")
    pdf.bullet("Corporate R&D: accumulates proprietary knowledge without SaaS exposure")
    pdf.bullet("Medical devices: learns patient patterns without HIPAA cloud compliance headaches")
    pdf.who_pays("Defense contractors, government agencies, investigative journalism, pharma")

    # --- Why Now ---
    pdf.ln(6)
    pdf.section_title("Why Now -- The Timing Advantage")
    pdf.body_text(
        "Several converging factors make this the right moment for weight-based persistent memory:"
    )
    pdf.ln(2)

    widths = [55, 125]
    pdf.table_row(["Factor", "Status"], widths, bold=True)
    pdf.table_row(["LoRA fine-tuning", "Mature, fast, cheap"], widths)
    pdf.table_row(["4-bit quantization", "Models fit on consumer hardware"], widths)
    pdf.table_row(["MLX / Apple Silicon", "Training on a laptop is real"], widths)
    pdf.table_row(["Open-weight models", "Llama, Mistral, Gemma -- modifiable"], widths)
    pdf.table_row(["Cloud GPU rental", "$2/hr for an H100"], widths)
    pdf.table_row(["RAG saturation", "Everyone has RAG -- it's commoditized"], widths)

    pdf.ln(4)
    pdf.body_text(
        "RAG is already commodity infrastructure. Every startup has it. Weight-based persistent "
        "memory is the next layer -- and almost nobody is building it because the engineering "
        "is genuinely hard."
    )

    # --- The Moat ---
    pdf.ln(4)
    pdf.section_title("The Technical Moat")
    pdf.body_text(
        "The hard engineering problems already solved in the Sleeping LLM system are the "
        "competitive moat. Anyone can fine-tune a model. Almost nobody has built a safe, "
        "automated, continuous learning loop that does not destroy the model."
    )
    pdf.ln(2)
    pdf.bullet("Learning rate calibration per model size -- empirically found the viable window (extremely narrow for small models)")
    pdf.bullet("Fact extraction vs raw training -- the insight that structured Q&A pairs are essential for memory formation")
    pdf.bullet("Fuse-to-temp validation -- preventing catastrophic failures from ever reaching production weights")
    pdf.bullet("Spaced repetition across sleep cycles -- progressive memory strengthening inspired by neuroscience")
    pdf.bullet("Identity preservation -- the model does not lose itself while learning new knowledge")

    # --- The Pitch ---
    pdf.ln(6)
    pdf.highlight_box("\"RAG gives an LLM access to information. We give it actual memory.\"")

    pdf.output("/Users/vbaranov/My-Apps/j/sleeping_llm_usecases.pdf")
    print("PDF generated: sleeping_llm_usecases.pdf")


if __name__ == "__main__":
    build_pdf()
