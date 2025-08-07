import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class VulnerabilityBreakdown(BaseModel):
    info: int = Field(..., ge=0, description="Number of info level vulnerabilities")
    low: int = Field(..., ge=0, description="Number of low severity vulnerabilities")
    medium: int = Field(..., ge=0, description="Number of medium severity vulnerabilities")
    high: int = Field(..., ge=0, description="Number of high severity vulnerabilities")
    critical: int = Field(..., ge=0, description="Number of critical vulnerabilities")
    @property
    def total_vulnerabilities(self) -> int:
        return self.info + self.low + self.medium + self.high + self.critical

class WAFStatus(BaseModel):
    protected: bool = Field(..., description="Is WAF enabled/protected?")
    detected_waf: str = Field(..., description="Detected WAF type")

class SubdomainFinding(BaseModel):
    name: str
    category: str
    risk_severity: str
    score: float

class SubdomainRiskInsights(BaseModel):
    subdomain: str
    status: str
    risk_score: float
    severity: str
    waf_status: WAFStatus
    vulnerability_breakdown: VulnerabilityBreakdown
    findings: List[SubdomainFinding]
    scan_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

class SubdomainSummaryParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        sections = {
            'waf_analysis': [],
            'findings_analysis': [],
            'immediate_actions': [],
            'recommendations': []
        }
        current_section = None
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if 'WAF ANALYSIS' in line.upper():
                current_section = 'waf_analysis'
            elif 'FINDINGS ANALYSIS' in line.upper():
                current_section = 'findings_analysis'
            elif 'IMMEDIATE ACTIONS' in line.upper():
                current_section = 'immediate_actions'
            elif 'RECOMMENDATIONS' in line.upper():
                current_section = 'recommendations'
            elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                if current_section:
                    bullet_text = line.lstrip('•-*').strip()
                    if bullet_text:
                        sections[current_section].append(bullet_text)
        return sections

class SubdomainRiskAgent:
    def __init__(self, groq_api_key: str = None, model_name: str = "llama3-8b-8192"):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly.")
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=model_name,
            temperature=0.2,
            max_tokens=2048
        )
        self.parser = SubdomainSummaryParser()
        self._setup_prompt()

    def _setup_prompt(self):
        self.prompt_template = PromptTemplate(
            input_variables=['subdomain_data'],
            template="""
You are a cybersecurity expert. Given the following subdomain security assessment data:
{subdomain_data}

Generate a detailed subdomain security report with the following sections in bullet points:

## WAF ANALYSIS
• [Analyze the WAF (Web Application Firewall) status for this subdomain. Clearly state if WAF is enabled or not, what type of WAF is detected, and what this means for the subdomain's security. If WAF is not present, explain the risks and recommend actions. Always provide at least 2 bullet points.]

## FINDINGS ANALYSIS
• [Summarize the key findings, their categories, and risk severity.]

## IMMEDIATE ACTIONS
• [List urgent, actionable items for this subdomain.]

## RECOMMENDATIONS
• [Provide long-term recommendations for improving subdomain security.]

Instructions:
- Use clear, business-friendly language.
- Explain technical terms when first mentioned.
- Connect findings to business impact and risk.
- Use specific numbers and metrics from the data.
- Make recommendations actionable.
"""
        )

    def _format_insights_for_prompt(self, insights: SubdomainRiskInsights) -> str:
        vuln = insights.vulnerability_breakdown
        findings_summary = '\n'.join([
            f"• {f.name} [{f.category}] - {f.risk_severity} (Score: {f.score})" for f in insights.findings
        ])
        formatted = f"""
SUBDOMAIN: {insights.subdomain}
STATUS: {insights.status}
RISK SCORE: {insights.risk_score}
SEVERITY: {insights.severity}
SCAN DATE: {insights.scan_date}

VULNERABILITY BREAKDOWN:
• Info: {vuln.info}
• Low: {vuln.low}
• Medium: {vuln.medium}
• High: {vuln.high}
• Critical: {vuln.critical}

WAF STATUS:
• Protected: {'Yes' if insights.waf_status.protected else 'No'}
• Detected WAF: {insights.waf_status.detected_waf}

FINDINGS:
{findings_summary if findings_summary else '• No findings detected'}
"""
        return formatted

    def generate_report(self, insights: SubdomainRiskInsights) -> dict:
        formatted = self._format_insights_for_prompt(insights)
        chain = self.prompt_template | self.llm | self.parser
        result = chain.invoke({"subdomain_data": formatted})
        return {
            "subdomain": insights.subdomain,
            "report": result,
            "generated_at": datetime.now().isoformat()
        }

def create_gunsnroses_sample_data():
    return SubdomainRiskInsights(
        subdomain="www.gunsnroses.com",
        status="Active",
        risk_score=0.69,
        severity="Low",
        waf_status=WAFStatus(
            protected=True,
            detected_waf="Generic"
        ),
        vulnerability_breakdown=VulnerabilityBreakdown(
            info=0,
            low=1,
            medium=0,
            high=0,
            critical=0
        ),
        findings=[
            SubdomainFinding(name="Weak Cipher Suites Detection", category="Vulnerabilities", risk_severity="Low", score=3.8),
            SubdomainFinding(name="HTTP Missing Security Headers", category="Vulnerabilities", risk_severity="Info", score=0.5),
            SubdomainFinding(name="TLS Version - Detect", category="Vulnerabilities", risk_severity="Info", score=0.5),
            SubdomainFinding(name="Deprecated TLS Detection", category="Vulnerabilities", risk_severity="Info", score=0.5),
            SubdomainFinding(name="Wappalyzer Technology Detection", category="Incident History", risk_severity="Info", score=0.5)
        ]
    )

def demo_subdomain_agent():
    agent = SubdomainRiskAgent()
    sample_insights = create_gunsnroses_sample_data()
    print("Generating subdomain security report for www.gunsnroses.com...")
    report = agent.generate_report(sample_insights)
    print(f"\n{'='*60}")
    print(f"SUBDOMAIN SECURITY REPORT - {report['subdomain'].upper()}")
    print(f"Generated: {report['generated_at']}")
    print(f"{'='*60}")
    # Only print the four relevant sections
    for section in ["waf_analysis", "findings_analysis", "immediate_actions", "recommendations"]:
        bullets = report['report'].get(section, [])
        print(f"\n## {section.replace('_', ' ').upper()}")
        for bullet in bullets:
            print(f"• {bullet}")
    print(f"\n{'='*60}")

if __name__ == "__main__":
    demo_subdomain_agent()
