"""Generate realistic sample documents (DOCX, TXT, PDF) into ./docs.

    .venv/bin/python scripts/make_docs.py [output_dir]

The files cover every path in the pipeline: native text, headings and lists,
tables, embedded images that need OCR, a fully scanned PDF page, page
headers/footers (noise to clean), and a non-UTF-8 text file.
All company names and data are fictional.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COMPANY = "Northwind Analytics"


# ----------------------------------------------------------------- image helpers
def font(size: int):
    return ImageFont.load_default(size=size)


def text_image(lines: list[str], size=(1600, 500), title: str | None = None) -> bytes:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y = 40
    if title:
        d.text((40, y), title, fill="black", font=font(52))
        y += 90
    for ln in lines:
        d.text((40, y), ln, fill="black", font=font(40))
        y += 64
    return _png(img)


def table_image(rows: list[list[str]], col_w=380, row_h=80) -> bytes:
    """A table drawn as a picture (like a screenshot pasted into a document)."""
    w, h = col_w * len(rows[0]) + 40, row_h * len(rows) + 40
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x0, y0 = 20 + c * col_w, 20 + r * row_h
            d.rectangle([x0, y0, x0 + col_w, y0 + row_h], outline="black", width=2)
            d.text((x0 + 16, y0 + 20), cell, fill="black", font=font(36))
    return _png(img)


def bar_chart_image() -> bytes:
    img = Image.new("RGB", (1600, 800), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 30), "Monthly Active Users by Quarter", fill="black", font=font(50))
    data = [("Q1", 42), ("Q2", 55), ("Q3", 61), ("Q4", 78)]
    for i, (q, v) in enumerate(data):
        x = 150 + i * 340
        d.rectangle([x, 700 - v * 7, x + 200, 700], fill=(70, 110, 180))
        d.text((x + 60, 720), q, fill="black", font=font(40))
        d.text((x + 40, 640 - v * 7), f"{v}k users", fill="black", font=font(36))
    return _png(img)


def _png(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


# ----------------------------------------------------------------------- DOCX
def make_employee_handbook(path: Path) -> None:
    import docx
    from docx.shared import Inches

    d = docx.Document()
    d.add_heading(f"{COMPANY} Employee Handbook", level=0)
    d.add_paragraph(
        f"Welcome to {COMPANY}. This handbook explains the policies, benefits and "
        "expectations that apply to every employee. It is reviewed every January by "
        "the People Operations team. If anything here conflicts with your employment "
        "contract, the contract takes precedence."
    )

    d.add_heading("1. Working Hours", level=1)
    d.add_paragraph(
        "Core hours are 10:00 to 16:00 in your local time zone. Outside core hours you "
        "may arrange your day flexibly, provided you complete 40 hours per week. "
        "Meetings should be scheduled within core hours whenever possible."
    )
    d.add_heading("Remote Work", level=2)
    d.add_paragraph(
        "Employees may work remotely up to three days per week. Fully remote "
        "arrangements require written approval from your manager and People Operations."
    )
    for item in ["Keep your calendar up to date with your working location.",
                 "Be reachable on chat during core hours.",
                 "Use the company VPN when accessing internal systems."]:
        d.add_paragraph(item, style="List Bullet")

    d.add_heading("2. Leave Policy", level=1)
    d.add_paragraph(
        "All full-time employees accrue paid leave from their first day. Leave requests "
        "must be submitted in the HR portal at least two weeks in advance for absences "
        "longer than three days. Unused annual leave of up to five days can be carried "
        "over to the following year; anything above five days expires on 31 March."
    )
    rows = [["Leave type", "Days per year", "Paid", "Notice required"],
            ["Annual leave", "25", "Yes", "2 weeks"],
            ["Sick leave", "12", "Yes", "Same day"],
            ["Parental leave", "90", "Yes", "8 weeks"],
            ["Bereavement", "5", "Yes", "None"],
            ["Unpaid leave", "Up to 30", "No", "4 weeks"]]
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            t.cell(r, c).text = val

    d.add_heading("3. Expenses", level=1)
    d.add_paragraph(
        "Business expenses are reimbursed within 15 working days of an approved claim. "
        "Receipts are mandatory for any expense above 25 USD. The limits below were "
        "approved by the Finance Committee and are shown as issued."
    )
    # A table pasted as an image: only reachable through OCR
    d.add_picture(io.BytesIO(table_image([
        ["Category", "Daily limit", "Approver"],
        ["Meals", "60 USD", "Manager"],
        ["Hotel", "220 USD", "Manager"],
        ["Flights", "Economy", "Finance"],
    ])), width=Inches(6))

    d.add_heading("4. Security", level=1)
    d.add_paragraph(
        "Laptops must use full-disk encryption and lock automatically after five "
        "minutes of inactivity. Report lost devices to the IT helpdesk within one hour. "
        "Never share your password, and enable multi-factor authentication on every "
        "company account."
    )
    d.add_picture(io.BytesIO(text_image(
        ["Phishing reports go to security@northwind.example",
         "Helpdesk is open 24 hours on extension 4400"],
        title="Security Contacts")), width=Inches(6))
    d.save(str(path))


def make_onboarding_checklist(path: Path) -> None:
    import docx

    d = docx.Document()
    d.add_heading("New Hire Onboarding Checklist", level=0)
    d.add_paragraph("Managers should complete this checklist with each new hire during "
                    "the first two weeks. Tick each item in the HR portal once done.")
    for week, items in {
        "Week 1": ["Collect laptop and badge from IT on day one.",
                   "Complete security and privacy training modules.",
                   "Meet your onboarding buddy and schedule a weekly check-in.",
                   "Set up payroll and benefits enrolment in the HR portal."],
        "Week 2": ["Shadow two customer calls with an account manager.",
                   "Read the product architecture overview.",
                   "Agree on 30-60-90 day goals with your manager."],
    }.items():
        d.add_heading(week, level=1)
        for it in items:
            d.add_paragraph(it, style="List Number")
    d.save(str(path))


# ------------------------------------------------------------------------ TXT
def make_txt_files(out: Path) -> None:
    (out / "meeting_notes_2026-11-10.txt").write_text(f"""\
{COMPANY} - Product Planning Meeting
Date: 10 November 2026
Attendees: Product, Engineering, Design, Customer Success

1. Q4 roadmap review
The team reviewed progress on the Q4 roadmap. The dashboard redesign is 80 percent
complete and on track for release on 1 December. The data export feature slipped by
two weeks because of an upstream API change from the billing provider.

2. Customer feedback
Customer Success reported that the most requested feature is scheduled email reports.
Twelve enterprise customers asked for it this quarter. Several customers also reported
slow load times on dashboards with more than fifty widgets.

3. Decisions
- Scheduled email reports move into the Q1 2027 roadmap as the top priority.
- Engineering will profile dashboard rendering and propose fixes by 24 November.
- Design will run five customer interviews about the new navigation.

4. Action items
- Priya: share the performance profiling plan by Friday.
- Marco: update the public roadmap page after the release date is confirmed.
- Lena: schedule customer interviews and circulate the discussion guide.

Next meeting: 24 November 2026.
""")

    (out / "product_faq.txt").write_text(f"""\
{COMPANY} Insights - Frequently Asked Questions

What is Northwind Insights?
Northwind Insights is a self-service analytics platform. It connects to your databases
and spreadsheets, and lets teams build dashboards without writing SQL.

Which data sources are supported?
PostgreSQL, MySQL, Snowflake, BigQuery, Amazon Redshift, Google Sheets and CSV uploads.
New connectors are released every quarter.

How is my data secured?
All data is encrypted in transit with TLS 1.3 and at rest with AES-256. Customer data is
stored in the region chosen at sign-up, and we never use customer data to train models.

How much does it cost?
The Starter plan is 15 USD per user per month and includes up to 10 dashboards.
The Business plan is 40 USD per user per month with unlimited dashboards and SSO.
Enterprise pricing is available on request and includes a dedicated support manager.

Can I cancel at any time?
Yes. Monthly plans can be cancelled at any time and remain active until the end of the
billing period. Annual plans can be cancelled with a pro-rated refund within 30 days.

How do I contact support?
Email support@northwind.example or use the in-app chat. Business and Enterprise
customers receive a response within four business hours.
""")

    # A file saved by an old Windows tool: cp1252, not UTF-8
    (out / "supplier_list_legacy.txt").write_bytes("""\
Approved Supplier List – updated by Procurement

Café Lumière Catering – office catering, Paris office, contract until June 2027.
Müller Büroservice GmbH – office furniture and supplies, Berlin office.
São Paulo Tech Distribuição – laptop leasing for the LATAM team.
Señor Limpieza S.L. – cleaning services, Madrid office.

All suppliers must hold a valid “Supplier Code of Conduct” certificate.
Renewals are handled by Procurement 60 days before the contract end date.
""".encode("cp1252"))


# ------------------------------------------------------------------------ PDF
def make_product_manual(path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)

    st = getSampleStyleSheet()
    H, P = st["Heading1"], st["BodyText"]
    story = [Paragraph("Northwind Insights User Manual", st["Title"]),
             Paragraph("Version 3.2 - November 2026", P), Spacer(1, 20)]

    story += [Paragraph("1. Getting Started", H), Paragraph(
        "Northwind Insights runs in any modern browser. After your administrator invites "
        "you, open the invitation email and choose a password of at least twelve "
        "characters. On first sign-in you are asked to enable multi-factor "
        "authentication with an authenticator app. The home screen lists the dashboards "
        "shared with you, recently viewed reports and any alerts that need attention.", P),
        Spacer(1, 10), Paragraph(
        "To create your first dashboard, select New Dashboard, choose a data source and "
        "drag fields onto the canvas. Charts update automatically as you add filters. "
        "Dashboards are saved every thirty seconds, and the version history keeps the "
        "last fifty versions so you can restore an earlier layout at any time.", P),
        PageBreak()]

    story += [Paragraph("2. Plans and Limits", H), Paragraph(
        "Each plan has limits on dashboards, data refresh frequency and storage. When "
        "you reach a limit, existing dashboards keep working but new ones cannot be "
        "created until you upgrade or archive old content.", P), Spacer(1, 12)]
    t = Table([["Plan", "Price / user / month", "Dashboards", "Refresh", "Storage"],
               ["Starter", "15 USD", "10", "Daily", "5 GB"],
               ["Business", "40 USD", "Unlimited", "Hourly", "100 GB"],
               ["Enterprise", "Custom", "Unlimited", "Every 5 minutes", "Unlimited"]],
              colWidths=[2.6 * cm, 3.8 * cm, 2.6 * cm, 3.2 * cm, 2.4 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    story += [t, Spacer(1, 16), Paragraph(
        "The chart below shows platform adoption over the past year. It is included as "
        "an image exported from the analytics dashboard.", P), Spacer(1, 8),
        RLImage(io.BytesIO(bar_chart_image()), width=15 * cm, height=7.5 * cm),
        PageBreak()]

    story += [Paragraph("3. Troubleshooting", H)]
    for q, a in [
        ("Dashboard loads slowly", "Dashboards with more than fifty widgets can take "
         "several seconds to render. Split large dashboards into tabs, or enable query "
         "caching under Settings, Performance."),
        ("Data is out of date", "Check the refresh schedule of the data source. On the "
         "Starter plan data refreshes once per day at 02:00 UTC. Select Refresh Now to "
         "trigger an immediate update."),
        ("Cannot sign in", "Reset your password from the sign-in page. If your "
         "authenticator device is lost, ask your administrator to reset multi-factor "
         "authentication for your account."),
    ]:
        story += [Paragraph(f"<b>{q}.</b> {a}", P), Spacer(1, 10)]

    def furniture(canvas, doc):  # header/footer noise the cleaner should strip
        canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, A4[1] - 1.2 * cm, f"{COMPANY} - Internal Use Only")
        canvas.drawString(2 * cm, 1 * cm, "Copyright 2026 Northwind Analytics")
        canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, f"Page {doc.page}")

    SimpleDocTemplate(str(path), pagesize=A4).build(story, onFirstPage=furniture,
                                                     onLaterPages=furniture)


def make_scanned_contract(path: Path) -> None:
    """Image-only PDF, like a contract that was printed, signed and scanned."""
    import pymupdf

    pages = [
        ("SERVICE AGREEMENT", [
            "This agreement is made on 3 March 2026 between",
            "Northwind Analytics Ltd and Contoso Retail Group.",
            "",
            "1. Services. Northwind will provide the Insights",
            "platform on the Business plan for 250 users.",
            "",
            "2. Term. The agreement runs for 24 months and renews",
            "automatically unless cancelled 60 days in advance.",
            "",
            "3. Fees. Contoso will pay 10,000 USD per month,",
            "invoiced monthly and payable within 30 days.",
        ]),
        ("SERVICE LEVELS AND SIGNATURES", [
            "4. Availability. Northwind guarantees 99.9 percent",
            "monthly uptime. Service credits of 10 percent apply",
            "for each full hour of downtime beyond this target.",
            "",
            "5. Confidentiality. Both parties keep the terms of",
            "this agreement confidential for five years.",
            "",
            "Signed for Northwind: J. Alvarez, Director",
            "Signed for Contoso: R. Okafor, Head of Procurement",
        ]),
    ]
    doc = pymupdf.open()
    for title, lines in pages:
        img = Image.open(io.BytesIO(text_image(lines, size=(1700, 1250), title=title)))
        img = img.convert("L").rotate(0.6, fillcolor=255)  # grayscale scan, slight skew
        page = doc.new_page(width=595, height=842)
        page.insert_image(pymupdf.Rect(30, 40, 565, 433), stream=_png(img))
    doc.save(str(path), deflate=True, garbage=3)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    make_employee_handbook(out / "employee_handbook.docx")
    make_onboarding_checklist(out / "onboarding_checklist.docx")
    make_txt_files(out)
    make_product_manual(out / "product_manual.pdf")
    make_scanned_contract(out / "scanned_service_agreement.pdf")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:34s} {f.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "docs"))
