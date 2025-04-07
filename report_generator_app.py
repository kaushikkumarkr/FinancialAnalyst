import streamlit as st
from utils.pdf_generator import generate_company_report_pdf
from chains.income_chain import income_chain
from chains.sec_rag_chain import sec_chain
from chains.news_chain import news_chain

st.set_page_config(page_title="📊 IntelliFin Report Generator")
st.title("📊 IntelliFin Report Generator")

company = st.text_input("Enter company ticker (e.g., TSLA):", "TSLA")
st.markdown("---")

# Section checkboxes
st.markdown("### Select sections to include in the report:")
include_income = st.checkbox("📈 Income Statement Summary", value=True)
include_sec = st.checkbox("📄 SEC Filing Insights", value=True)
include_news = st.checkbox("📰 Recent News Summary", value=True)

st.markdown("---")

if st.button("🧠 Generate Report"):
    with st.spinner("Running AI chains to gather report data..."):
        data = {}

        if include_income:
            data["income"] = income_chain()

        if include_sec:
            data["sec"] = sec_chain()

        if include_news:
            data["news"] = news_chain()

        if not data:
            st.error("Please select at least one section to generate the report.")
        else:
            file_path = generate_company_report_pdf(company, data)
            with open(file_path, "rb") as f:
                st.success("✅ Report generated!")
                st.download_button("📥 Download PDF", f, file_name=f"{company}_report.pdf")
