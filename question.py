import os
from typing import List, Dict
from enum import Enum
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


class VendorSize(str, Enum):
    """Vendor size categories"""
    STARTUP = "Startup (1-50 employees)"
    SMALL = "Small (51-200 employees)"
    MEDIUM = "Medium (201-1000 employees)"
    LARGE = "Large (1001-5000 employees)"
    ENTERPRISE = "Enterprise (5000+ employees)"


class Industry(str, Enum):
    """Industry categories"""
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


class QuestionType(str, Enum):
    """Types of assessment questions"""
    SINGLE_CHOICE = "single_choice"
    TEXT = "text"


class Question(BaseModel):
    """Individual assessment question"""
    question_id: str = Field(..., description="Unique identifier for the question")
    question_text: str = Field(..., description="The actual question to ask the vendor")
    question_type: QuestionType = Field(..., description="Type of question (single_choice or text)")
    question_weight: float = Field(..., ge=1, le=10, description="Weight/importance of the question (1-10)")
    rationale: str = Field(..., description="Why this question is important for risk assessment")
    expected_answer: str = Field(..., description="What constitutes a low-risk answer")
    red_flags: List[str] = Field(..., description="Answers or responses that would indicate high risk")
    evidence_required: bool = Field(..., description="Whether document evidence is required for this question")


class QuestionCategory(BaseModel):
    """Category of related questions"""
    category_name: str = Field(..., description="Name of the question category")
    category_description: str = Field(..., description="Description of what this category covers")
    questions: List[Question] = Field(..., description="List of questions in this category")


class VendorInput(BaseModel):
    """Input model for vendor assessment"""
    template_name: str = Field(..., description="Name of the assessment template")
    industry: Industry = Field(..., description="Industry sector of the vendor")
    vendor_size: VendorSize = Field(..., description="Size category of the vendor")
    vendor_description: str = Field(..., description="Detailed description of vendor services and capabilities")


class VendorAssessmentQuestionnaire(BaseModel):
    """Complete vendor risk assessment questionnaire"""
    assessment_overview: str = Field(..., description="Overview of the assessment approach")
    categories: List[QuestionCategory] = Field(..., description="List of question categories")
    
    @property
    def total_questions(self) -> int:
        """Calculate total number of questions across all categories"""
        return sum(len(category.questions) for category in self.categories)


class VendorQuestionnaireAgent:
    """Third-party vendor risk assessment questionnaire generator"""
    
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly.")
        
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=8000
        )
        
        self.output_parser = PydanticOutputParser(pydantic_object=VendorAssessmentQuestionnaire)
        self.prompt_template = self._create_prompt_template()
        self.chain = self._create_chain()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_template("""
You are an expert in third-party risk management specializing in vendor assessment questionnaires.

VENDOR DETAILS:
- Template Name: {template_name}
- Industry: {industry}
- Vendor Size: {vendor_size}
- Vendor Description: {vendor_description}

Generate a comprehensive vendor risk assessment questionnaire. Return a JSON object with this structure:

{{
  "assessment_overview": "Comprehensive vendor risk assessment covering security, compliance, and operational controls",
  "categories": [
    {{
      "category_name": "Framework Compliance & Governance",
      "category_description": "NIST/ISO alignment and framework compliance",
      "questions": [
        {{
          "question_id": "Q1",
          "question_text": "Does the vendor have a documented framework compliance program?",
          "question_type": "single_choice",
          "question_weight": 8,
          "rationale": "Ensures the vendor has a structured approach to compliance",
          "expected_answer": "Yes",
          "red_flags": ["No"],
          "evidence_required": true
        }}
      ]
    }}
  ]
}}

CRITICAL REQUIREMENTS:
1. **Dynamic Question Count**: Each category should have 3-15 questions based on relevance
2. **Skip Irrelevant Categories**: Only include categories relevant to the vendor's services and industry
3. **Question Weight**: Each question weight must be between 1-10 based on importance
4. **Evidence Requirements**: Set evidence_required to true for questions that need documentation (policies, procedures, certificates, audit reports, etc.)
5. **Category Selection**: Choose from these categories based on vendor relevance:
   - Framework Compliance & Governance (for regulated industries)
   - Information Security & Cybersecurity (for all vendors)
   - Risk Management & Assessment (for all vendors)
   - Access Control & Identity Management (for vendors handling sensitive data)
   - Data Protection & Cryptography (for vendors processing data)
   - Incident Response & Business Continuity (for critical service providers)
   - Audit & Compliance Monitoring (for regulated industries)
   - Third-Party & Supply Chain Management (for vendors with their own suppliers)
   - Physical & Environmental Security (for vendors with physical infrastructure)
   - Human Resources & Training (for vendors with employees)
   - System & Network Security (for technology vendors)
   - Industry-Specific Requirements (based on vendor industry)

6. **Vendor-Specific Focus**:
   - Technology vendors: Focus on cybersecurity, data protection, system security
   - Financial services: Emphasize compliance, risk management, audit
   - Healthcare: Prioritize HIPAA compliance, data protection, privacy
   - Manufacturing: Focus on supply chain, operational continuity, physical security
   - Small vendors: Fewer categories, basic security questions
   - Large vendors: More comprehensive coverage across categories

7. **JSON Format**: Return ONLY the JSON object, no additional text or explanations
8. **Variable Substitution**: Use the exact template variable names: {template_name}, {industry}, {vendor_size}, {vendor_description}
""")
        
        return prompt
    
    def _create_chain(self):
        chain = (
            {
                "template_name": RunnablePassthrough(),
                "industry": RunnablePassthrough(),
                "vendor_size": RunnablePassthrough(),
                "vendor_description": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
            | self.output_parser
        )
        
        return chain
    
    def generate_questionnaire(self, vendor_input: VendorInput) -> VendorAssessmentQuestionnaire:
        try:
            input_data = {
                "template_name": vendor_input.template_name,
                "industry": vendor_input.industry.value,
                "vendor_size": vendor_input.vendor_size.value,
                "vendor_description": vendor_input.vendor_description
            }
            
            questionnaire = self.chain.invoke(input_data)
            return questionnaire
            
        except Exception as e:
            raise Exception(f"Error generating questionnaire: {str(e)}")
    
    def generate_questionnaire_dict(self, vendor_data: dict) -> VendorAssessmentQuestionnaire:
        try:
            vendor_input = VendorInput(**vendor_data)
            return self.generate_questionnaire(vendor_input)
        except Exception as e:
            raise Exception(f"Error creating questionnaire from dictionary: {str(e)}")


def example_usage():
    agent = VendorQuestionnaireAgent()
    
    vendor_scenarios = [
        {
            "template_name": "Cloud Service Provider Assessment",
            "industry": "Technology",
            "vendor_size": "Large (1001-5000 employees)",
            "vendor_description": "Cloud infrastructure provider offering IaaS, PaaS, and SaaS solutions including data storage, computing resources, AI/ML services, and enterprise applications. Serves Fortune 500 companies with global data centers."
        },
        {
            "template_name": "Small Healthcare Vendor Assessment",
            "industry": "Healthcare",
            "vendor_size": "Small (51-200 employees)",
            "vendor_description": "Small healthcare software company providing patient management systems for local clinics. Handles patient data and medical records."
        }
    ]
    
    for i, vendor_scenario in enumerate(vendor_scenarios, 1):
        print(f"\n{'='*80}")
        print(f"VENDOR QUESTIONNAIRE {i}: {vendor_scenario['template_name']}")
        print(f"{'='*80}")
        
        try:
            questionnaire = agent.generate_questionnaire_dict(vendor_scenario)
            
            print(f"\nAssessment Overview:\n{questionnaire.assessment_overview}")
            print(f"\nTotal Questions: {questionnaire.total_questions}")
            
            print(f"\nQUESTION CATEGORIES ({len(questionnaire.categories)}):")
            print("-" * 50)
            
            for j, category in enumerate(questionnaire.categories, 1):
                print(f"\n{j}. {category.category_name}")
                print(f"   Description: {category.category_description}")
                print(f"   Questions: {len(category.questions)}")
                
                for k, question in enumerate(category.questions[:2], 1):
                    print(f"\n   Sample Question {k}:")
                    print(f"   Q{question.question_id}: {question.question_text}")
                    print(f"   Type: {question.question_type} | Weight: {question.question_weight:.1f}")
                    print(f"   Expected Answer: {question.expected_answer}")
                    print(f"   Evidence Required: {'Yes' if question.evidence_required else 'No'}")
                    if question.red_flags:
                        print(f"   Red Flags: {', '.join(question.red_flags[:2])}...")
                
                if len(category.questions) > 2:
                    print(f"   ... and {len(category.questions) - 2} more questions")
            
        except Exception as e:
            print(f"Error processing scenario: {e}")
        
        if i < len(vendor_scenarios):
            print(f"\n{'-'*80}")
            print("Continuing to next vendor...")
            print()


if __name__ == "__main__":
    example_usage()