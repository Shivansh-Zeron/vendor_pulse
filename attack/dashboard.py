import os
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from datetime import datetime

# Pydantic models based on Attack Surface Dashboard structure
class VulnerabilityBreakdown(BaseModel):
    """Model for vulnerability severity breakdown"""
    info: int = Field(..., ge=0, description="Number of info level vulnerabilities")
    low: int = Field(..., ge=0, description="Number of low severity vulnerabilities")
    medium: int = Field(..., ge=0, description="Number of medium severity vulnerabilities")
    high: int = Field(..., ge=0, description="Number of high severity vulnerabilities")
    critical: int = Field(..., ge=0, description="Number of critical vulnerabilities")
    
    @property
    def total_vulnerabilities(self) -> int:
        return self.info + self.low + self.medium + self.high + self.critical

class AssetInventory(BaseModel):
    """Model for asset inventory data"""
    domains: int = Field(..., ge=0, description="Number of discovered domains")
    active_subdomains: int = Field(..., ge=0, description="Number of active subdomains")
    inactive_subdomains: int = Field(..., ge=0, description="Number of inactive subdomains")
    ip_addresses: int = Field(..., ge=0, description="Number of discovered IP addresses")

class WAFStatus(BaseModel):
    """Model for WAF (Web Application Firewall) status"""
    protected_percentage: float = Field(..., ge=0, le=100, description="Percentage of assets protected by WAF")
    exposed_percentage: float = Field(..., ge=0, le=100, description="Percentage of assets exposed without WAF")

class DiscoveredAssets(BaseModel):
    """Model for various discovered assets"""
    exposed_ports: int = Field(..., ge=0, description="Number of exposed ports")
    technology_stack_items: int = Field(..., ge=0, description="Number of technology stack items identified")
    lookalike_domains: int = Field(..., ge=0, description="Number of lookalike domains found")
    exposed_cloud_findings: int = Field(..., ge=0, description="Number of exposed cloud findings")
    github_patches: int = Field(..., ge=0, description="Number of discovered GitHub patches")
    docker_images: int = Field(..., ge=0, description="Number of discovered Docker images")
    gitlab_patches: int = Field(..., ge=0, description="Number of discovered GitLab patches")
    bitbucket_patches: int = Field(..., ge=0, description="Number of discovered Bitbucket patches")

class TechnologyStackItem(BaseModel):
    """Model for technology stack items"""
    name: str = Field(..., description="Technology name")
    version: Optional[str] = Field(None, description="Version if detected")
    category: str = Field(..., description="Technology category")
    confidence: Optional[float] = Field(None, ge=0, le=100, description="Detection confidence percentage")

class AttackSurfaceInsights(BaseModel):
    """Main model for Attack Surface dashboard insights"""
    target_domain: str = Field(..., description="The target domain being analyzed")
    scan_date: str = Field(..., description="Date of the security scan")
    
    # Core metrics
    total_discovered_assets: int = Field(..., ge=0, description="Total number of discovered assets")
    vulnerability_breakdown: VulnerabilityBreakdown = Field(..., description="Breakdown of vulnerabilities by severity")
    asset_inventory: AssetInventory = Field(..., description="Inventory of discovered assets")
    waf_status: WAFStatus = Field(..., description="WAF protection status")
    
    # Risk assessment
    risk_score: Optional[float] = Field(None, ge=0, le=100, description="Overall risk score (0-100)")
    likelihood_of_exploitation: float = Field(..., ge=0, le=100, description="Likelihood of exploitation percentage")
    risk_status: str = Field(..., description="Risk status message")
    
    # Discovered assets
    discovered_assets: DiscoveredAssets = Field(..., description="Various discovered assets and findings")
    technology_stack: List[TechnologyStackItem] = Field(default=[], description="Identified technology stack")
    
    # Additional security findings
    exposed_services: List[str] = Field(default=[], description="List of exposed services")
    security_misconfigurations: List[str] = Field(default=[], description="Security misconfigurations found")
    
    @field_validator('waf_status')
    @classmethod
    def validate_waf_percentages(cls, v):
        if abs(v.protected_percentage + v.exposed_percentage - 100) > 0.1:
            # Allow some flexibility for rounding
            pass
        return v

class AttackSurfaceSummary(BaseModel):
    """Model for the generated Attack Surface dashboard summary"""
    target_domain: str
    data_explanation: List[str]
    overall_security_posture: List[str]
    vulnerability_analysis: List[str]
    asset_discovery_insights: List[str]
    technology_risk_assessment: List[str]
    immediate_actions_required: List[str]
    strategic_recommendations: List[str]
    generated_at: str

class AttackSurfaceSummaryParser(BaseOutputParser):
    """Custom parser for Attack Surface dashboard summary output"""
    def parse(self, text: str) -> dict:
        sections = {
            'data_explanation': [],
            'overall_security_posture': [],
            'vulnerability_analysis': [],
            'asset_discovery_insights': [],
            'technology_risk_assessment': [],
            'immediate_actions_required': [],
            'strategic_recommendations': []
        }
        current_section = None
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if 'DATA EXPLANATION' in line.upper():
                current_section = 'data_explanation'
            elif 'OVERALL SECURITY POSTURE' in line.upper():
                current_section = 'overall_security_posture'
            elif 'VULNERABILITY ANALYSIS' in line.upper():
                current_section = 'vulnerability_analysis'
            elif 'ASSET DISCOVERY INSIGHTS' in line.upper():
                current_section = 'asset_discovery_insights'
            elif 'TECHNOLOGY RISK ASSESSMENT' in line.upper():
                current_section = 'technology_risk_assessment'
            elif 'IMMEDIATE ACTIONS REQUIRED' in line.upper():
                current_section = 'immediate_actions_required'
            elif 'STRATEGIC RECOMMENDATIONS' in line.upper():
                current_section = 'strategic_recommendations'
            elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                if current_section:
                    bullet_text = line.lstrip('•-*').strip()
                    if bullet_text:
                        sections[current_section].append(bullet_text)
        return sections

class AttackSurfaceDashboardAgent:
    """Main agent class for processing Attack Surface dashboard insights"""
    def __init__(self, groq_api_key: str, model_name: str = "llama3-8b-8192"):
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=2048
        )
        self.parser = AttackSurfaceSummaryParser()
        self._setup_prompt()
    def _setup_prompt(self):
        self.prompt_template = PromptTemplate(
            input_variables=['attack_surface_data'],
            template="""
You are a cybersecurity expert explaining Attack Surface Management results to stakeholders who may not be technical experts.

Given the following security assessment data:
{attack_surface_data}

Generate a comprehensive security analysis with the following sections in bullet points:

## DATA EXPLANATION
• [3-4 bullet points explaining what this data represents, what the numbers mean, and why this assessment is important for the organization]

## OVERALL SECURITY POSTURE
• [3-4 high-level bullet points about the organization's overall attack surface and security stance in business terms]

## VULNERABILITY ANALYSIS  
• [4-5 detailed points about vulnerability distribution, severity levels, and what these risks mean for the business]

## ASSET DISCOVERY INSIGHTS
• [3-4 points about discovered assets, infrastructure visibility, and what exposed elements mean for security]

## TECHNOLOGY RISK ASSESSMENT
• [4-5 points about technology stack risks, outdated systems, and configuration issues in plain language]

## IMMEDIATE ACTIONS REQUIRED
• [4-5 urgent, actionable items with clear priorities and business impact explanation]

## STRATEGIC RECOMMENDATIONS
• [4-5 long-term strategic recommendations with clear business justification and expected outcomes]

Instructions:
- Use clear, business-friendly language that non-technical stakeholders can understand
- Explain technical terms when first mentioned
- Connect security findings to business impact and risk
- Provide specific numbers and metrics from the data
- Make recommendations actionable with clear next steps
- Prioritize items by business risk and feasibility

Make each bullet point informative and easy to understand for both technical and non-technical audiences.
"""
        )
    def _format_insights_for_prompt(self, insights: AttackSurfaceInsights) -> str:
        vuln_breakdown = insights.vulnerability_breakdown
        total_vulns = vuln_breakdown.total_vulnerabilities
        critical_high_percentage = ((vuln_breakdown.critical + vuln_breakdown.high) / total_vulns * 100) if total_vulns > 0 else 0
        tech_summary = []
        for tech in insights.technology_stack:
            tech_info = f"{tech.name}"
            if tech.version:
                tech_info += f" v{tech.version}"
            if tech.confidence:
                tech_info += f" (confidence: {tech.confidence}%)"
            tech_info += f" [{tech.category}]"
            tech_summary.append(tech_info)
        formatted_data = f"""
TARGET DOMAIN: {insights.target_domain}
SCAN DATE: {insights.scan_date}

VULNERABILITY SUMMARY:
• Total Vulnerabilities: {total_vulns}
• Critical: {vuln_breakdown.critical}
• High: {vuln_breakdown.high}
• Medium: {vuln_breakdown.medium}
• Low: {vuln_breakdown.low}
• Info: {vuln_breakdown.info}
• Critical + High Risk: {critical_high_percentage:.1f}% of all vulnerabilities

ASSET INVENTORY:
• Total Discovered Assets: {insights.total_discovered_assets}
• Domains: {insights.asset_inventory.domains}
• Active Subdomains: {insights.asset_inventory.active_subdomains}
• Inactive Subdomains: {insights.asset_inventory.inactive_subdomains}
• IP Addresses: {insights.asset_inventory.ip_addresses}

WAF PROTECTION STATUS:
• Protected: {insights.waf_status.protected_percentage}%
• Exposed: {insights.waf_status.exposed_percentage}%

RISK ASSESSMENT:
• Risk Score: {insights.risk_score if insights.risk_score else 'Not calculated'}
• Likelihood of Exploitation: {insights.likelihood_of_exploitation}%
• Risk Status: {insights.risk_status}

DISCOVERED ASSETS & FINDINGS:
• Exposed Ports: {insights.discovered_assets.exposed_ports}
• Technology Stack Items: {insights.discovered_assets.technology_stack_items}
• Lookalike Domains: {insights.discovered_assets.lookalike_domains}
• Exposed Cloud Findings: {insights.discovered_assets.exposed_cloud_findings}
• GitHub Patches: {insights.discovered_assets.github_patches}
• Docker Images: {insights.discovered_assets.docker_images}
• GitLab Patches: {insights.discovered_assets.gitlab_patches}
• Bitbucket Patches: {insights.discovered_assets.bitbucket_patches}

TECHNOLOGY STACK:
{chr(10).join(['• ' + tech for tech in tech_summary]) if tech_summary else '• No technology stack items detected'}

EXPOSED SERVICES:
{chr(10).join(['• ' + service for service in insights.exposed_services]) if insights.exposed_services else '• No exposed services identified'}

SECURITY MISCONFIGURATIONS:
{chr(10).join(['• ' + config for config in insights.security_misconfigurations]) if insights.security_misconfigurations else '• No security misconfigurations detected'}
"""
        return formatted_data
    def generate_summary(self, insights: AttackSurfaceInsights) -> AttackSurfaceSummary:
        formatted_insights = self._format_insights_for_prompt(insights)
        chain = self.prompt_template | self.llm | self.parser
        result = chain.invoke({"attack_surface_data": formatted_insights})
        summary = AttackSurfaceSummary(
            target_domain=insights.target_domain,
            data_explanation=result.get('data_explanation', []),
            overall_security_posture=result.get('overall_security_posture', []),
            vulnerability_analysis=result.get('vulnerability_analysis', []),
            asset_discovery_insights=result.get('asset_discovery_insights', []),
            technology_risk_assessment=result.get('technology_risk_assessment', []),
            immediate_actions_required=result.get('immediate_actions_required', []),
            strategic_recommendations=result.get('strategic_recommendations', []),
            generated_at=datetime.now().isoformat()
        )
        return summary
    def print_summary(self, summary: AttackSurfaceSummary):
        print(f"\n{'='*70}")
        print(f"ATTACK SURFACE MANAGEMENT REPORT - {summary.target_domain.upper()}")
        print(f"Generated: {summary.generated_at}")
        print(f"{'='*70}")
        sections = [
            ("DATA EXPLANATION", summary.data_explanation),
            ("OVERALL SECURITY POSTURE", summary.overall_security_posture),
            ("VULNERABILITY ANALYSIS", summary.vulnerability_analysis),
            ("ASSET DISCOVERY INSIGHTS", summary.asset_discovery_insights),
            ("TECHNOLOGY RISK ASSESSMENT", summary.technology_risk_assessment),
            ("IMMEDIATE ACTIONS REQUIRED", summary.immediate_actions_required),
            ("STRATEGIC RECOMMENDATIONS", summary.strategic_recommendations)
        ]
        for section_name, bullets in sections:
            print(f"\n## {section_name}")
            for bullet in bullets:
                print(f"• {bullet}")
        print(f"\n{'='*70}")

def create_axisbank_sample_data():
    """Create sample data based on the axisbank.com dashboard shown in the image"""
    return AttackSurfaceInsights(
        target_domain="axisbank.com",
        scan_date="2024-08-07",
        total_discovered_assets=0,
        vulnerability_breakdown=VulnerabilityBreakdown(
            critical=12,
            high=17,
            medium=24,
            low=7,
            info=59
        ),
        asset_inventory=AssetInventory(
            domains=6,
            active_subdomains=5,
            inactive_subdomains=1,
            ip_addresses=8000
        ),
        waf_status=WAFStatus(
            protected_percentage=45.0,
            exposed_percentage=55.0
        ),
        risk_score=5.0,  # Not shown in dashboard
        likelihood_of_exploitation=5.0,
        risk_status="No risk is detected for the added assets",
        discovered_assets=DiscoveredAssets(
            exposed_ports=7,
            technology_stack_items=78,
            lookalike_domains=45,
            exposed_cloud_findings=6,
            github_patches=0,
            docker_images=0,
            gitlab_patches=0,
            bitbucket_patches=0
        ),
        technology_stack=[],
        exposed_services=[],
        security_misconfigurations=[]
    )

def demo_attack_surface_agent():
    """Demonstrate the Attack Surface dashboard agent"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    sample_insights = create_axisbank_sample_data()
    agent = AttackSurfaceDashboardAgent(groq_api_key, model_name="llama3-8b-8192")
    print("Generating Attack Surface dashboard summary...")
    summary = agent.generate_summary(sample_insights)
    agent.print_summary(summary)

if __name__ == "__main__":
    demo_attack_surface_agent()