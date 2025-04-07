import os
import markdown
import pdfkit
from datetime import datetime


def generate_company_report(
    company_name: str,
    selected_sections: dict,
    output_dir: str = "generated_reports/"
) -> str:
    """
    Generate a markdown-based financial report and convert it to PDF.

    Args:
        company_name (str): The name of the company.
        selected_sections (dict): Dictionary of section_name -> content.
        output_dir (str): Folder to save the PDF.

    Returns:
        str: Path to the generated PDF file.
    """
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{company_name.replace(' ', '_')}_report_{now}.pdf"
    pdf_path = os.path.join(output_dir, filename)

    # Generate markdown content
    markdown_lines = [f"# {company_name} Financial Report\n"]
    for section, content in selected_sections.items():
        markdown_lines.append(f"\n## {section}\n")
        markdown_lines.append(content.strip() if content.strip() else "_No data available._")

    markdown_text = "\n".join(markdown_lines)
    html = markdown.markdown(markdown_text)

    # Convert HTML to PDF using pdfkit (make sure wkhtmltopdf is installed)
    try:
        pdfkit.from_string(html, pdf_path)
    except OSError as e:
        raise RuntimeError("PDF generation failed. Is wkhtmltopdf installed?") from e

    return pdf_path
