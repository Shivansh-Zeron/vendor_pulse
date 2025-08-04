import os
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


class VendorIndustry(str, Enum):
    """Vendor industry categories"""
    TECHNOLOGY = "Technology"
    FINANCIAL_SERVICES = "Financial Services"
    HEALTHCARE = "Healthcare"
    MANUFACTURING = "Manufacturing"
    RETAIL = "Retail"
    TELECOMMUNICATIONS = "Telecommunications"
    ENERGY = "Energy"
    TRANSPORTATION = "Transportation"
    CONSULTING = "Consulting"
    EDUCATION = "Education"
    GOVERNMENT = "Government"
    OTHER = "Other"


class VendorCategory(str, Enum):
    """Vendor service categories"""
    CLOUD_SERVICES = "Cloud Services"
    SOFTWARE_DEVELOPMENT = "Software Development"
    IT_SERVICES = "IT Services"
    DATA_PROCESSING = "Data Processing"
    PAYMENT_PROCESSING = "Payment Processing"
    CONSULTING_SERVICES = "Consulting Services"
    INFRASTRUCTURE = "Infrastructure"
    SECURITY_SERVICES = "Security Services"
    ANALYTICS = "Analytics"
    COMMUNICATION = "Communication"
    OTHER = "Other"


class VulnerabilityCounts(BaseModel):
    """Count of vulnerabilities by severity level"""
    low: int = Field(..., ge=0, description="Number of low severity vulnerabilities")
    medium: int = Field(..., ge=0, description="Number of medium severity vulnerabilities")
    high: int = Field(..., ge=0, description="Number of high severity vulnerabilities")
    critical: int = Field(..., ge=0, description="Number of critical severity vulnerabilities")
    
    @property
    def total_vulnerabilities(self) -> int:
        """Calculate total number of vulnerabilities"""
        return self.low + self.medium + self.high + self.critical


class VendorRiskInput(BaseModel):
    """Input model for vendor risk assessment"""
    vendor_name: str = Field(..., description="Name of the vendor")
    vendor_details: str = Field(..., description="Detailed description of vendor services and capabilities")
    vendor_size: int = Field(..., ge=1, description="Number of employees at the vendor")
    vendor_industry: VendorIndustry = Field(..., description="Industry sector of the vendor")
    vendor_category: VendorCategory = Field(..., description="Service category of the vendor")
    likelihood_of_exploitation: float = Field(..., ge=0.0, le=1.0, description="Likelihood of exploitation level (0.0-1.0)")
    total_vulnerabilities: VulnerabilityCounts = Field(..., description="Total vulnerabilities categorized by severity")
    risk_score: float = Field(..., ge=0, le=10, description="Overall risk score (0-10, where 10 is highest risk)")


class RiskAssessmentResult(BaseModel):
    """Individual risk assessment result"""
    risk_category: str = Field(..., description="Category of risk assessed")
    risk_level: str = Field(..., description="Assessed risk level (Low/Medium/High/Critical)")
    risk_score: float = Field(..., ge=0, le=10, description="Numerical risk score for this category")
    impact_description: str = Field(..., description="Description of potential impact")
    mitigation_recommendations: List[str] = Field(..., description="Specific mitigation recommendations")
    monitoring_requirements: str = Field(..., description="Ongoing monitoring requirements")


class VendorRiskAssessmentResponse(BaseModel):
    """Complete vendor risk assessment response"""
    vendor_summary: Dict[str, str] = Field(..., description="Summary of vendor information")
    overall_risk_rating: str = Field(..., description="Overall risk rating (Low/Medium/High/Critical)")
    overall_risk_score: float = Field(..., ge=0, le=10, description="Overall numerical risk score")
    vulnerability_analysis: Dict[str, Any] = Field(..., description="Analysis of vulnerability distribution")
    exploitation_assessment: str = Field(..., description="Assessment of exploitation likelihood and impact")
    risk_categories: List[RiskAssessmentResult] = Field(..., description="Detailed risk assessment by category")
    executive_summary: str = Field(..., description="Executive summary of key findings")
    immediate_actions: List[str] = Field(..., description="Immediate actions required")
    long_term_recommendations: List[str] = Field(..., description="Long-term risk management recommendations")
    compliance_considerations: List[str] = Field(..., description="Relevant compliance and regulatory considerations")


class VendorRiskAssessmentAgent:
    """AI agent for comprehensive vendor risk assessment"""
    
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly.")
        
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=8000
        )
        
        self.output_parser = PydanticOutputParser(pydantic_object=VendorRiskAssessmentResponse)
        self.prompt_template = self._create_prompt_template()
        self.chain = self._create_chain()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_template("""
You are a world-class expert in cybersecurity risk assessment and third-party vendor risk management with over 20 years of experience. 
You specialize in analyzing vendor security postures, vulnerability assessments, and providing comprehensive risk evaluations 
for enterprise organizations.

Your expertise includes:
- Cybersecurity risk assessment and threat modeling
- Vulnerability management and exploit analysis
- Third-party vendor risk evaluation
- Industry-specific security requirements and compliance
- Risk quantification and scoring methodologies
- Incident response and business impact assessment
- Regulatory compliance (SOX, GDPR, HIPAA, PCI-DSS, SOC 2, ISO 27001)
- Supply chain security and vendor management

VENDOR RISK ASSESSMENT DATA:

**Vendor Information:**
- Vendor Name: {vendor_name}
- Vendor Details: {vendor_details}
- Vendor Size: {vendor_size} employees
- Industry: {vendor_industry}
- Service Category: {vendor_category}

**Risk Metrics:**
- Likelihood of Exploitation: {likelihood_of_exploitation} (probability: 0.0-1.0)
- Vulnerability Counts:
  * Critical: {critical_vulns}
  * High: {high_vulns}
  * Medium: {medium_vulns}
  * Low: {low_vulns}
  * Total: {total_vulns}
- Overall Risk Score: {risk_score}/10

ASSESSMENT INSTRUCTIONS:

Conduct a comprehensive vendor risk assessment based on the provided data. Your analysis should consider:

**1. VULNERABILITY ANALYSIS:**
- Analyze the distribution and severity of vulnerabilities
- Assess the potential impact of each vulnerability category
- Consider the vendor's industry and size in vulnerability context
- Evaluate the correlation between vulnerability counts and exploitation likelihood

**2. EXPLOITATION LIKELIHOOD ASSESSMENT:**
- Analyze the likelihood of exploitation based on vendor characteristics
- Consider industry-specific threat landscapes
- Assess attacker motivation and capability requirements
- Evaluate vendor's exposure and attack surface

**3. RISK CATEGORIZATION:**
Assess risks across these categories:
- **Information Security Risk**: Data protection, access controls, security architecture
- **Operational Risk**: Service availability, business continuity, performance
- **Compliance Risk**: Regulatory adherence, audit requirements, legal obligations
- **Financial Risk**: Vendor stability, contract terms, liability exposure
- **Reputational Risk**: Brand impact, customer trust, market perception
- **Supply Chain Risk**: Dependencies, third-party relationships, vendor ecosystem
- **Technology Risk**: System integration, compatibility, technical debt

**4. INDUSTRY-SPECIFIC CONSIDERATIONS:**
- Technology: Focus on IP protection, scalability, innovation risk
- Financial Services: Emphasize regulatory compliance, data protection, systemic risk
- Healthcare: HIPAA compliance, patient safety, data privacy
- Manufacturing: Operational continuity, supply chain disruption, safety
- Government: Security clearance, classified data, national security

**5. SIZE-BASED RISK FACTORS:**
- Small (1-50): Financial stability, resource constraints, security maturity
- Medium (51-200): Growth scalability, process maturity, compliance readiness
- Large (201-1000): Complex governance, multiple touchpoints, systemic impact
- Enterprise (1000+): Extensive infrastructure, regulatory complexity, global impact

**6. RISK SCORING METHODOLOGY:**
- Use 0-10 scale where 10 represents maximum risk
- Consider probability and impact in scoring
- Weight critical vulnerabilities heavily
- Factor in industry and size-specific risk multipliers

**7. MITIGATION AND RECOMMENDATIONS:**
- Provide specific, actionable mitigation strategies
- Prioritize recommendations based on risk severity
- Consider cost-benefit analysis in recommendations
- Address both immediate and long-term risk management

Generate a comprehensive risk assessment that provides clear insights for executive decision-making 
and actionable recommendations for risk mitigation.

CRITICAL: Return ONLY the JSON object with the required fields. Do not include any additional text, 
explanations, or schema definitions. The response must be a valid JSON object that matches the 
expected structure exactly.

Example JSON structure:
{{
  "vendor_summary": {{
    "name": "Vendor Name",
    "size": "10 employees",
    "industry": "Manufacturing"
  }},
  "overall_risk_rating": "Medium",
  "overall_risk_score": 5.5,
  "vulnerability_analysis": {{
    "total_vulnerabilities": 15,
    "critical_count": 1,
    "high_count": 1
  }},
  "exploitation_assessment": "Assessment details here",
  "risk_categories": [
    {{
      "risk_category": "Information Security Risk",
      "risk_level": "Medium",
      "risk_score": 6.0,
      "impact_description": "Impact description",
      "mitigation_recommendations": ["Recommendation 1", "Recommendation 2"],
      "monitoring_requirements": "Monitoring details"
    }}
  ],
  "executive_summary": "Executive summary here",
  "immediate_actions": ["Action 1", "Action 2"],
  "long_term_recommendations": ["Recommendation 1", "Recommendation 2"],
  "compliance_considerations": ["Consideration 1", "Consideration 2"]
}}

{format_instructions}

Conduct the vendor risk assessment:
""")
        
        return prompt
    
    def _create_chain(self):
        chain = (
            {
                "vendor_name": RunnablePassthrough(),
                "vendor_details": RunnablePassthrough(),
                "vendor_size": RunnablePassthrough(),
                "vendor_industry": RunnablePassthrough(),
                "vendor_category": RunnablePassthrough(),
                "likelihood_of_exploitation": RunnablePassthrough(),
                "critical_vulns": RunnablePassthrough(),
                "high_vulns": RunnablePassthrough(),
                "medium_vulns": RunnablePassthrough(),
                "low_vulns": RunnablePassthrough(),
                "total_vulns": RunnablePassthrough(),
                "risk_score": RunnablePassthrough(),
                "format_instructions": lambda _: self.output_parser.get_format_instructions()
            }
            | self.prompt_template
            | self.llm
            | self.output_parser
        )
        
        return chain
    
    def assess_vendor_risk(self, vendor_input: VendorRiskInput) -> VendorRiskAssessmentResponse:
        try:
            # Prepare input data for the chain
            input_data = {
                "vendor_name": vendor_input.vendor_name,
                "vendor_details": vendor_input.vendor_details,
                "vendor_size": vendor_input.vendor_size,
                "vendor_industry": vendor_input.vendor_industry.value,
                "vendor_category": vendor_input.vendor_category.value,
                "likelihood_of_exploitation": vendor_input.likelihood_of_exploitation,
                "critical_vulns": vendor_input.total_vulnerabilities.critical,
                "high_vulns": vendor_input.total_vulnerabilities.high,
                "medium_vulns": vendor_input.total_vulnerabilities.medium,
                "low_vulns": vendor_input.total_vulnerabilities.low,
                "total_vulns": vendor_input.total_vulnerabilities.total_vulnerabilities,
                "risk_score": vendor_input.risk_score
            }
            
            # Process through the chain
            assessment = self.chain.invoke(input_data)
            
            return assessment
            
        except Exception as e:
            raise Exception(f"Error conducting vendor risk assessment: {str(e)}")
    
    def assess_vendor_risk_dict(self, vendor_data: dict) -> VendorRiskAssessmentResponse:
        try:
            vendor_input = VendorRiskInput(**vendor_data)
            return self.assess_vendor_risk(vendor_input)
        except Exception as e:
            raise Exception(f"Error creating assessment from dictionary: {str(e)}")


# Example usage and testing
def example_usage():
    """Example of how to use the VendorRiskAssessmentAgent"""
    
    # Initialize the agent
    agent = VendorRiskAssessmentAgent()
    
    # Example vendor risk scenarios
    vendor_scenarios = [
        {
            "vendor_name": "Hindalco Everlast",
            "vendor_details": "Hindalco Everlast is a brand under Hindalco Industries Limited, part of the Aditya Birla Group. Hindalco is one of Asia's largest producers of primary aluminium and a global leader in copper production. The company is involved in various sectors, including building and construction, where its products are widely used. Hindalco Everlast specializes in aluminium roofing solutions, offering a range of products such as Hi Crest Sheets, Circular Corrugated Sheets, Troughed Sheets, Modern Toughed Sheets, Tiled Profile Sheets, and roofing structurals. The brand caters to both residential and business customers, providing high-quality materials designed for durability and security in roofing applications.",
            "vendor_size": 10,  # Small size
            "vendor_industry": "Manufacturing",
            "vendor_category": "Infrastructure",
            "likelihood_of_exploitation": 0.3,  # Medium
            "total_vulnerabilities": {
                "low": 1,
                "medium": 12,
                "high": 1,
                "critical": 1
            },
            "risk_score": 2.2
        },
        {
            "vendor_name": "FinTech Solutions Ltd.",
            "vendor_details": "Financial technology company providing payment processing, fraud detection, and regulatory compliance solutions for banking institutions. Handles PCI-compliant payment data.",
            "vendor_size": 2500,  # Large size
            "vendor_industry": "Financial Services",
            "vendor_category": "Payment Processing",
            "likelihood_of_exploitation": 0.5,  # Medium
            "total_vulnerabilities": {
                "low": 10,
                "medium": 5,
                "high": 2,
                "critical": 0
            },
            "risk_score": 4.2
        },
        {
            "vendor_name": "HealthData Analytics",
            "vendor_details": "Healthcare analytics startup providing AI-powered patient data analysis and clinical decision support tools. Processes PHI and medical records for hospital systems.",
            "vendor_size": 25,  # Startup size
            "vendor_industry": "Healthcare",
            "vendor_category": "Analytics",
            "likelihood_of_exploitation": 0.9,  # Very High
            "total_vulnerabilities": {
                "low": 5,
                "medium": 12,
                "high": 8,
                "critical": 3
            },
            "risk_score": 8.7
        }
    ]
    
    # Process each vendor scenario
    for i, scenario in enumerate(vendor_scenarios, 1):
        print(f"\n{'='*100}")
        print(f"VENDOR RISK ASSESSMENT {i}: {scenario['vendor_name']}")
        print(f"{'='*100}")
        
        try:
            # Conduct the risk assessment
            assessment = agent.assess_vendor_risk_dict(scenario)
            
            # Display executive summary
            print(f"\n📊 EXECUTIVE SUMMARY")
            print(f"{'='*50}")
            print(f"Overall Risk Rating: {assessment.overall_risk_rating}")
            print(f"Overall Risk Score: {assessment.overall_risk_score:.1f}/10")
            print(f"\nSummary: {assessment.executive_summary}")
            
            # Display vulnerability analysis
            print(f"\n🔍 VULNERABILITY ANALYSIS")
            print(f"{'='*50}")
            for key, value in assessment.vulnerability_analysis.items():
                print(f"{key}: {value}")
            
            # Display exploitation assessment
            print(f"\n⚠️ EXPLOITATION ASSESSMENT")
            print(f"{'='*50}")
            print(assessment.exploitation_assessment)
            
            # Display risk categories
            print(f"\n📋 RISK CATEGORIES ASSESSMENT")
            print(f"{'='*50}")
            for risk_cat in assessment.risk_categories[:3]:  # Show first 3 categories
                print(f"\n{risk_cat.risk_category}: {risk_cat.risk_level} (Score: {risk_cat.risk_score:.1f})")
                print(f"Impact: {risk_cat.impact_description}")
                print(f"Top Mitigation: {risk_cat.mitigation_recommendations[0] if risk_cat.mitigation_recommendations else 'None'}")
            
            if len(assessment.risk_categories) > 3:
                print(f"\n... and {len(assessment.risk_categories) - 3} more risk categories assessed")
            
            # Display immediate actions
            print(f"\n🚨 IMMEDIATE ACTIONS REQUIRED")
            print(f"{'='*50}")
            for action in assessment.immediate_actions[:3]:  # Show first 3 actions
                print(f"• {action}")
            
            if len(assessment.immediate_actions) > 3:
                print(f"• ... and {len(assessment.immediate_actions) - 3} more actions")
            
            # Display compliance considerations
            print(f"\n📜 COMPLIANCE CONSIDERATIONS")
            print(f"{'='*50}")
            for compliance in assessment.compliance_considerations[:2]:  # Show first 2
                print(f"• {compliance}")
            
        except Exception as e:
            print(f"Error processing scenario: {e}")


if __name__ == "__main__":
    example_usage()