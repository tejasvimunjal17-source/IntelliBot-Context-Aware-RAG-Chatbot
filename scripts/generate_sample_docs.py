"""Create the sample knowledge-base files in data/sample_docs/.

All content is FICTIONAL (a made-up company called NovaTech Solutions) so the
chatbot cannot know these facts from its training data. That makes it easy to
test whether answers really come from the documents.

Run:  python scripts/generate_sample_docs.py      (needs: pip install reportlab)
The generated files are already included in the project, so you only need to
run this if you want to change the content.
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"

# NOTE: plain ASCII only (the PDF built-in fonts have no rupee symbol), so
# money is written as "INR".

PDF_DOCS = {
    "NovaTech_Leave_and_Attendance_Policy.pdf": (
        "NovaTech Solutions - Leave and Attendance Policy",
        [
            ("1. Working Hours", [
                "Standard working hours at NovaTech Solutions are 9:30 AM to 6:00 PM, Monday to Friday. "
                "Core hours, when every employee must be available for meetings, are 11:00 AM to 4:00 PM.",
                "Employees may work remotely for up to 2 days per week with written approval from their manager. "
                "Remote work on more than 2 days per week requires approval from the HR Director.",
            ]),
            ("2. Annual Leave", [
                "Every full-time employee receives 24 days of paid annual leave per calendar year. Leave is "
                "earned at the rate of 2 days per month. Up to 10 unused days can be carried forward to the "
                "next year. Any days above 10 expire on 31 March.",
                "Leave of more than 3 consecutive days must be requested at least 14 days in advance. "
                "All leave requests are submitted through the NovaHR portal.",
            ]),
            ("3. Sick Leave and Casual Leave", [
                "Employees receive 12 days of paid sick leave per year. A medical certificate is required "
                "if sick leave lasts more than 2 consecutive days.",
                "Employees also receive 6 days of casual leave per year for urgent personal matters. Casual "
                "leave cannot be carried forward.",
            ]),
            ("4. Parental Leave", [
                "Maternity leave is 26 weeks of paid leave. Paternity leave is 10 working days of paid leave "
                "and must be taken within 6 months of the child's birth or adoption.",
            ]),
            ("5. Public Holidays", [
                "NovaTech Solutions observes 12 public holidays each year. The full list is published on the "
                "NovaHR portal every January. If a public holiday falls on a weekend, no replacement day is given.",
            ]),
        ],
    ),
    "NovaTech_IT_Security_Policy.pdf": (
        "NovaTech Solutions - IT Security Policy",
        [
            ("1. Passwords and Access", [
                "All passwords must be at least 14 characters long and must be changed every 180 days. "
                "Multi-factor authentication (MFA) is mandatory for email, the NovaHR portal, and all "
                "cloud systems. Sharing passwords with anyone, including IT staff, is not allowed.",
            ]),
            ("2. Devices", [
                "Every company laptop has full-disk encryption enabled and must lock automatically after "
                "5 minutes of inactivity. USB storage devices are blocked by default. An employee who needs "
                "USB access must raise a ticket with the IT Helpdesk and get written approval from the "
                "Security Team.",
                "Installing software that is not on the approved list requires an approved IT ticket. "
                "Employees must never install software from unknown websites.",
            ]),
            ("3. Remote Access", [
                "When working outside the office, employees must connect through NovaVPN before opening any "
                "internal system. Public Wi-Fi may only be used together with NovaVPN.",
            ]),
            ("4. Data Classification", [
                "NovaTech data is classified into four levels: Public, Internal, Confidential, and Restricted. "
                "Restricted data, such as customer payment details, must never be sent by email or stored on "
                "personal devices.",
            ]),
            ("5. Reporting Incidents", [
                "Any suspected security incident, such as a lost laptop or a phishing email that was opened, "
                "must be reported to security@novatech.example within 1 hour. Suspicious emails should be "
                "reported using the Report Phishing button in the email client. Employees who report "
                "quickly will not be penalized.",
            ]),
        ],
    ),
    "NovaDesk_Product_FAQ.pdf": (
        "NovaDesk - Product FAQ",
        [
            ("About NovaDesk", [
                "NovaDesk is a cloud-based customer support helpdesk sold by NovaTech Solutions. It lets teams "
                "manage customer tickets from email, chat, and web forms in one place.",
            ]),
            ("Pricing and Plans", [
                "NovaDesk has three plans. The Starter plan costs USD 15 per agent per month. The Business plan "
                "costs USD 39 per agent per month. The Enterprise plan has custom pricing and requires a "
                "minimum of 50 agents. All plans include a free trial of 21 days, and no credit card is needed "
                "to start the trial.",
                "Refunds are available within 30 days of the first purchase. After 30 days, payments are "
                "non-refundable.",
            ]),
            ("Reliability and Data", [
                "NovaDesk guarantees 99.9 percent uptime for Business and Enterprise customers. Customer data "
                "is stored in data centers in Mumbai and Frankfurt. Customers can choose their data region "
                "when they sign up. All data can be exported in CSV format at any time.",
            ]),
            ("Support", [
                "Starter customers get email support on weekdays. Business customers get 24x5 chat and email "
                "support (24 hours a day, Monday to Friday). Enterprise customers get 24x7 support and a "
                "dedicated account manager.",
            ]),
            ("Integrations and API", [
                "NovaDesk integrates with Slack, Microsoft Teams, and Jira. The API rate limit is 100 "
                "requests per minute on the Starter plan and 1,000 requests per minute on the Business plan. "
                "Enterprise customers can request custom limits.",
            ]),
        ],
    ),
    "NovaTech_Expense_Reimbursement_Policy.pdf": (
        "NovaTech Solutions - Expense Reimbursement Policy",
        [
            ("1. Submitting Claims", [
                "Expense claims must be submitted within 30 days of the expense date. Claims submitted after "
                "30 days will be rejected. Receipts are required for every expense above INR 500.",
                "Approved claims are paid within 10 working days. A manager can approve claims up to "
                "INR 25,000. Claims above INR 25,000 also need approval from the department head.",
            ]),
            ("2. Travel", [
                "Flights must be booked in economy class for journeys shorter than 6 hours. Business class "
                "is allowed only for flights of 6 hours or longer and needs approval from the department head. "
                "Using a personal car for official travel is reimbursed at INR 12 per kilometre.",
            ]),
            ("3. Hotels and Meals", [
                "The hotel limit is INR 6,500 per night in metro cities (Mumbai, Delhi, Bengaluru, Chennai, "
                "Kolkata, Hyderabad) and INR 4,500 per night in all other cities. The daily meal allowance "
                "during domestic travel is INR 1,200.",
            ]),
            ("4. Not Reimbursed", [
                "Personal expenses, alcohol, fines for traffic violations, and entertainment that is not "
                "approved in advance are not reimbursed.",
            ]),
        ],
    ),
}

TXT_DOCS = {
    "NovaTech_Employee_Onboarding_Guide.txt": """NovaTech Solutions - Employee Onboarding Guide

WELCOME
Welcome to NovaTech Solutions! This guide explains what to expect in your first months.

FIRST DAY
Report to the reception desk at 9:30 AM. You will receive your company laptop, an ID card,
and your NovaHR portal login. Every new employee is assigned a buddy, an experienced colleague
who helps you settle in during the first 3 months.

PROBATION
The probation period is 6 months. A review meeting with your manager takes place at the end of
month 3 and again at the end of month 6. After a successful review, your employment is confirmed
in writing.

THE 30-60-90 DAY PLAN
Days 1 to 30: Learn the company, meet your team, and complete the mandatory security training.
Days 31 to 60: Start working on real tasks with support from your buddy.
Days 61 to 90: Work independently and agree on goals for the next 6 months with your manager.

LEARNING BUDGET
Every employee gets a learning budget of INR 30,000 per year for courses, books, and certifications.
Courses must be related to your role and approved by your manager before purchase.

WHO TO CONTACT
IT Helpdesk: extension 1100, or helpdesk@novatech.example
HR Team: hr@novatech.example
Security Team: security@novatech.example
The cafeteria on the 2nd floor is open from 8:30 AM to 4:00 PM on working days.
""",
}


def build_pdf(path: Path, title: str, sections) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, title=title, author="NovaTech Solutions (fictional)",
        leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
    )
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 12)]
    for heading, paragraphs in sections:
        story.append(Paragraph(escape(heading), styles["Heading2"]))
        for text in paragraphs:
            story.append(Paragraph(escape(text), styles["BodyText"]))
            story.append(Spacer(1, 6))
    story.append(Spacer(1, 18))
    story.append(Paragraph(
        "<i>Sample document with fictional content, created for the IntelliBot demo.</i>",
        styles["Italic"],
    ))
    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (title, sections) in PDF_DOCS.items():
        build_pdf(OUT_DIR / filename, title, sections)
        print("created", filename)
    for filename, content in TXT_DOCS.items():
        (OUT_DIR / filename).write_text(content, encoding="utf-8")
        print("created", filename)


if __name__ == "__main__":
    main()
