import os
from fpdf import FPDF
import unicodedata

REPORT_DIR = "pdf_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def clean_text(text):
    # Normalize and remove characters not supported by latin-1
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", "ignore").decode("latin-1")

class CompanyReportPDF(FPDF):
    def __init__(self, company):
        super().__init__()
        self.company = clean_text(company)
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_title(f"{self.company} Report")

    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, f"{self.company} - AI-Powered Financial Report", ln=True, align="C")
        self.ln(5)

    def section_title(self, title):
        self.set_font("Arial", "B", 12)
        self.set_text_color(0)
        self.cell(0, 10, clean_text(title), ln=True)
        self.ln(1)

    def section_body(self, text):
        self.set_font("Arial", "", 11)
        self.set_text_color(50)
        self.multi_cell(0, 6, clean_text(text))
        self.ln()

def generate_company_report_pdf(company: str, data: dict) -> str:
    pdf = CompanyReportPDF(company)
    
    for section_title, section_content in data.items():
        pdf.section_title(section_title)
        pdf.section_body(section_content)

    output_path = os.path.join(REPORT_DIR, f"{clean_text(company)}_report.pdf")
    pdf.output(output_path)
    return output_path
