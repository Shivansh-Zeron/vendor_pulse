import os
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, validator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


class RiskPriority(str, Enum):
    """Risk priority levels"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ResponseStrategy(str, Enum):
    """Risk response strategies"""
    AVOID = "Avoid"
    MITIGATE = "Mitigate"
    TRANSFER = "Transfer"
    ACCEPT = "Accept"
    MONITOR = "Monitor"


class RiskInput(BaseModel):
    """Input model for risk assessment"""
    risk_title: str = Field(..., description="Title of the identified risk")
    risk_description: str = Field(..., description="Detailed description of the risk")
    risk_scoring: float = Field(..., ge=0, le=10, description="Risk score from 0-10 (10 being highest risk)")
    risk_priority: RiskPriority = Field(..., description="Priority level of the risk")

    @validator('risk_scoring')
    def validate_risk_scoring(cls, v):
        if not 0 <= v <= 10:
            raise ValueError('Risk scoring must be between 0 and 10')
        return v


class Control(BaseModel):
    """Individual control measure"""
    control_id: str = Field(..., description="Unique identifier for the control")
    control_description: str = Field(..., description="Description of the control measure")
    control_type: str = Field(..., description="Type of control (Preventive, Detective, Corrective)")
    implementation_effort: str = Field(..., description="Implementation effort (Low, Medium, High)")
    effectiveness: str = Field(..., description="Control effectiveness (Low, Medium, High)")


class RiskResponse(BaseModel):
    """Output model for risk management response"""
    response_strategy: ResponseStrategy = Field(..., description="Primary response strategy for the risk")
    response_part: str = Field(..., description="Detailed response plan and actions to be taken")
    controls: List[Control] = Field(..., description="List of recommended control measures")


class VendorRiskManagementAgent:
    """Third-party vendor risk management agent"""
    
    def __init__(self, groq_api_key: str = None):
        """
        Initialize the risk management agent
        
        Args:
            groq_api_key: Groq API key (if not provided, will look for GROQ_API_KEY env var)
        """
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly.")
        
        # Initialize the Groq LLM
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.1-8b-instant",  # You can change this to other Groq models
            temperature=0.1,  # Low temperature for consistent risk assessment
            max_tokens=4000
        )
        
        # Set up the output parser
        self.output_parser = PydanticOutputParser(pydantic_object=RiskResponse)
        
        # Create the prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create the chain
        self.chain = self._create_chain()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for risk assessment"""
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert in third-party risk management and vendor risk management with over 15 years of experience. 
You specialize in assessing vendor risks, developing response strategies, and implementing control frameworks.

Your expertise includes:
- Third-party vendor due diligence
- Supply chain risk assessment
- Vendor performance monitoring
- Regulatory compliance (SOX, GDPR, SOC 2, ISO 27001)
- Business continuity planning
- Information security risk assessment
- Financial risk evaluation
- Operational risk management

Given the following risk details, provide a comprehensive risk management response:

RISK DETAILS:
- Risk Title: {risk_title}
- Risk Description: {risk_description}
- Risk Score: {risk_scoring}/10
- Risk Priority: {risk_priority}

INSTRUCTIONS:
1. Analyze the risk based on your expertise in vendor risk management
2. Consider the risk score and priority level
3. Determine the most appropriate response strategy (Avoid, Mitigate, Transfer, Accept, Monitor)
4. Develop a detailed response plan
5. Recommend specific, actionable controls
6. Consider industry best practices and regulatory requirements
7. Provide practical implementation guidance

Focus on:
- Vendor-specific risk factors
- Third-party relationship management
- Supply chain vulnerabilities
- Data privacy and security concerns
- Business continuity impacts
- Regulatory compliance requirements
- Financial and reputational risks

{format_instructions}

Provide your expert assessment and recommendations:
""")
        
        return prompt
    
    def _create_chain(self):
        """Create the LangChain processing chain"""
        
        chain = (
            {
                "risk_title": RunnablePassthrough(),
                "risk_description": RunnablePassthrough(),
                "risk_scoring": RunnablePassthrough(),
                "risk_priority": RunnablePassthrough(),
                "format_instructions": lambda _: self.output_parser.get_format_instructions()
            }
            | self.prompt_template
            | self.llm
            | self.output_parser
        )
        
        return chain
    
    def assess_risk(self, risk_input: RiskInput) -> RiskResponse:
        """
        Assess a vendor risk and provide management recommendations
        
        Args:
            risk_input: RiskInput object containing risk details
            
        Returns:
            RiskResponse object with strategy and controls
        """
        try:
            # Prepare input data
            input_data = {
                "risk_title": risk_input.risk_title,
                "risk_description": risk_input.risk_description,
                "risk_scoring": risk_input.risk_scoring,
                "risk_priority": risk_input.risk_priority.value
            }
            
            # Process through the chain
            response = self.chain.invoke(input_data)
            
            return response
            
        except Exception as e:
            raise Exception(f"Error processing risk assessment: {str(e)}")
    



# Example usage and testing
def example_usage():
    """Example of how to use the VendorRiskManagementAgent"""
    
    # Initialize the agent (make sure to set your GROQ_API_KEY)
    agent = VendorRiskManagementAgent()
    
    # Example risk scenarios
    risk_scenarios = [
        {
            "risk_title": "Cloud Service Provider Data Breach",
            "risk_description": "Our primary cloud storage vendor has experienced multiple security incidents in the past year, with the latest involving potential exposure of customer PII data. The vendor processes and stores sensitive customer information including financial records.",
            "risk_scoring": 8.5,
            "risk_priority": "Critical"
        },
        {
            "risk_title": "Payment Processor Service Disruption",
            "risk_description": "Third-party payment processor shows signs of financial instability and has had intermittent service outages affecting transaction processing during peak business hours.",
            "risk_scoring": 7.0,
            "risk_priority": "High"
        },
        {
            "risk_title": "Software Vendor License Compliance",
            "risk_description": "Uncertainty around software licensing compliance for critical business applications. Vendor has changed licensing terms and audit requirements.",
            "risk_scoring": 5.5,
            "risk_priority": "Medium"
        }
    ]
    
    # Process each risk scenario
    for i, scenario in enumerate(risk_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"RISK SCENARIO {i}: {scenario['risk_title']}")
        print(f"{'='*60}")
        
        try:
            # Create RiskInput object and assess the risk
            risk_input = RiskInput(**scenario)
            response = agent.assess_risk(risk_input)
            
            # Display results
            print(f"Response Strategy: {response.response_strategy}")
            print(f"\nResponse Plan:\n{response.response_part}")
            
            print(f"\nRecommended Controls ({len(response.controls)}):")
            for j, control in enumerate(response.controls, 1):
                print(f"{j}. {control.control_id} - {control.control_description}")
                print(f"   Type: {control.control_type} | Effort: {control.implementation_effort} | Effectiveness: {control.effectiveness}")
            
        except Exception as e:
            print(f"Error processing scenario: {e}")


if __name__ == "__main__":
    # Set your Groq API key here or in environment variable
    # os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
    
    example_usage()