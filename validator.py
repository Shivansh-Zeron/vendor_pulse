import os
import json
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Import the existing models from other files
from answer import VendorAnswer  # Expected answers
from vendor import PDFAssessmentAnswer  # Vendor PDF answers

load_dotenv()


class QuestionData(BaseModel):
    """Question data from question.py"""
    question: str = Field(..., description="The assessment question")
    question_type: str = Field(..., description="Type of question")
    question_weight: float = Field(..., description="Weight/importance of the question")
    category: str = Field(..., description="Category the question belongs to")


class ValidationInput(BaseModel):
    """Input model for vendor validation"""
    questions: List[QuestionData] = Field(..., description="Original assessment questions")
    expected_answers: List[VendorAnswer] = Field(..., description="Expected/ideal vendor answers")
    vendor_pdf_answers: List[PDFAssessmentAnswer] = Field(..., description="Actual vendor answers from PDF")


class VendorValidationResponse(BaseModel):
    """Complete vendor validation response"""
    detailed_summary: str = Field(..., description="Detailed risk assessment summary in bullet points")


class VendorValidatorAgent:
    """Vendor risk assessment validator agent"""
    
    def __init__(self, groq_api_key: str = None):
        """
        Initialize the validator agent
        
        Args:
            groq_api_key: Groq API key (if not provided, will look for GROQ_API_KEY env var)
        """
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly.")
        
        # Initialize the Groq LLM
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.1,  # Low temperature for analytical assessment
            max_tokens=8000
        )
        
        # Set up the output parser
        self.output_parser = StrOutputParser()
        
        # Create the prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create the chain
        self.chain = self._create_chain()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for vendor validation"""
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert third-party risk management analyst with over 20 years of experience in vendor due diligence, 
compliance assessment, and cybersecurity risk evaluation. You specialize in analyzing vendor responses against 
industry best practices and identifying potential security, operational, and compliance vulnerabilities.

Your expertise includes:
- Vendor risk assessment and due diligence
- Information security and cybersecurity analysis
- Compliance frameworks (ISO 27001, SOC 2, GDPR, HIPAA, PCI-DSS)
- Business continuity and operational risk evaluation
- Financial stability and vendor viability assessment
- Third-party risk management best practices
- Vulnerability identification and risk prioritization

TASK: Analyze the vendor's actual responses against expected best practice answers and provide a comprehensive 
risk assessment with detailed findings.

ASSESSMENT DATA:

ORIGINAL QUESTIONS:
{questions_json}

EXPECTED BEST PRACTICE ANSWERS:
{expected_answers_json}

VENDOR'S ACTUAL ANSWERS FROM PDF:
{vendor_answers_json}

ANALYSIS INSTRUCTIONS:

Conduct a thorough comparison analysis and provide a detailed risk assessment summary in bullet point format covering:

**COMPLIANCE GAPS & VULNERABILITIES:**
• Compare vendor answers against expected best practices
• Identify specific compliance gaps (ISO 27001, SOC 2, GDPR, etc.)
• Highlight missing security controls or inadequate implementations
• Flag answers marked as "None" (information not provided)

**RISK CATEGORIZATION:**
• **Critical Risks**: Issues that pose immediate threats to security, compliance, or operations
• **High Risks**: Significant gaps that require immediate attention and remediation
• **Medium Risks**: Important concerns that should be addressed within defined timeframes
• **Low Risks**: Minor issues or areas for improvement

**SECURITY POSTURE ASSESSMENT:**
• Evaluate information security maturity and controls
• Assess access control, encryption, and data protection measures
• Review incident response and business continuity capabilities
• Analyze vendor's security architecture and practices

**OPERATIONAL RISK EVALUATION:**
• Assess business continuity and disaster recovery readiness
• Evaluate financial stability indicators
• Review operational maturity and service delivery capabilities
• Identify potential single points of failure

**REGULATORY COMPLIANCE STATUS:**
• Assess compliance with relevant industry regulations
• Identify potential regulatory violations or gaps
• Evaluate documentation and audit trail adequacy
• Review data privacy and protection compliance

**VENDOR RELIABILITY INDICATORS:**
• Assess completeness and quality of vendor responses
• Evaluate transparency and willingness to provide information
• Identify evasive or incomplete answers
• Review evidence of mature risk management practices

**RECOMMENDATIONS & NEXT STEPS:**
• Immediate actions required before vendor engagement
• Due diligence requirements and additional documentation needed
• Contract terms and risk mitigation measures to implement
• Ongoing monitoring and review requirements

**OVERALL RISK RATING:**
• Provide an overall risk assessment (Low/Medium/High/Critical)
• Justify the rating based on findings
• Recommend proceed/proceed with conditions/do not proceed

Focus on being specific, actionable, and comprehensive. Use bullet points throughout for clarity and readability.
Prioritize findings by risk level and business impact.

Generate the detailed vendor risk assessment summary:
""")
        
        return prompt
    
    def _create_chain(self):
        """Create the LangChain processing chain"""
        
        chain = (
            {
                "questions_json": RunnablePassthrough(),
                "expected_answers_json": RunnablePassthrough(),
                "vendor_answers_json": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
            | self.output_parser
        )
        
        return chain
    
    def _load_questions_from_json(self, json_file_path: str) -> List[QuestionData]:
        """
        Load questions from JSON file
        
        Args:
            json_file_path: Path to the JSON file containing questions
            
        Returns:
            List of QuestionData objects
        """
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            questions = []
            for category, category_questions in data.items():
                for q in category_questions:
                    question_data = QuestionData(
                        question=q["question"],
                        question_type=q["question_type"],
                        question_weight=q["question_weight"],
                        category=category
                    )
                    questions.append(question_data)
            
            return questions
            
        except Exception as e:
            raise Exception(f"Error loading questions from JSON: {str(e)}")
    
    def validate_vendor_assessment(self, 
                                 questions_json_path: str,
                                 expected_answers: List[VendorAnswer],
                                 vendor_pdf_answers: List[PDFAssessmentAnswer]) -> str:
        """
        Validate vendor assessment by comparing expected vs actual answers
        
        Args:
            questions_json_path: Path to JSON file with original questions
            expected_answers: List of expected/ideal answers
            vendor_pdf_answers: List of vendor's actual answers from PDF
            
        Returns:
            Detailed risk assessment summary as string
        """
        try:
            # Load original questions
            questions = self._load_questions_from_json(questions_json_path)
            
            # Convert to JSON format for the prompt
            questions_json = json.dumps([
                {
                    "question": q.question,
                    "question_type": q.question_type,
                    "question_weight": q.question_weight,
                    "category": q.category
                } for q in questions
            ], indent=2)
            
            expected_answers_json = json.dumps([
                {
                    "question": ans.question,
                    "answer": ans.answer,
                    "category": ans.question_category
                } for ans in expected_answers
            ], indent=2)
            
            vendor_answers_json = json.dumps([
                {
                    "question": ans.question,
                    "answer": ans.answer
                } for ans in vendor_pdf_answers
            ], indent=2)
            
            # Prepare input data
            input_data = {
                "questions_json": questions_json,
                "expected_answers_json": expected_answers_json,
                "vendor_answers_json": vendor_answers_json
            }
            
            # Process through the chain
            detailed_summary = self.chain.invoke(input_data)
            
            return detailed_summary
            
        except Exception as e:
            raise Exception(f"Error validating vendor assessment: {str(e)}")
    
    def validate_from_objects(self,
                            questions: List[QuestionData],
                            expected_answers: List[VendorAnswer],
                            vendor_pdf_answers: List[PDFAssessmentAnswer]) -> str:
        """
        Validate vendor assessment from object inputs
        
        Args:
            questions: List of original questions
            expected_answers: List of expected/ideal answers
            vendor_pdf_answers: List of vendor's actual answers from PDF
            
        Returns:
            Detailed risk assessment summary as string
        """
        try:
            # Convert to JSON format for the prompt
            questions_json = json.dumps([
                {
                    "question": q.question,
                    "question_type": q.question_type,
                    "question_weight": q.question_weight,
                    "category": q.category
                } for q in questions
            ], indent=2)
            
            expected_answers_json = json.dumps([
                {
                    "question": ans.question,
                    "answer": ans.answer,
                    "category": ans.question_category
                } for ans in expected_answers
            ], indent=2)
            
            vendor_answers_json = json.dumps([
                {
                    "question": ans.question,
                    "answer": ans.answer
                } for ans in vendor_pdf_answers
            ], indent=2)
            
            # Prepare input data
            input_data = {
                "questions_json": questions_json,
                "expected_answers_json": expected_answers_json,
                "vendor_answers_json": vendor_answers_json
            }
            
            # Process through the chain
            detailed_summary = self.chain.invoke(input_data)
            
            return detailed_summary
            
        except Exception as e:
            raise Exception(f"Error validating vendor assessment: {str(e)}")


# Example usage demonstrating the complete workflow
def example_usage():
    """Example of complete vendor validation workflow"""
    
    # Initialize agents
    from answer import VendorAnswerAgent
    from vendor import PDFVendorAssessmentAgent
    
    answer_agent = VendorAnswerAgent()
    pdf_agent = PDFVendorAssessmentAgent()
    validator_agent = VendorValidatorAgent()
    
    try:
        print("Step 1: Generating expected answers...")
        # Generate expected answers
        expected_response = answer_agent.answer_questions_from_json("vendor_risk_categories_detailed.json")
        expected_answers = expected_response.answers
        
        print("Step 2: Analyzing vendor PDF...")
        # Analyze vendor PDF
        vendor_pdf_answers = pdf_agent.assess_pdf_with_questions(
            pdf_file_path="vendor_document.pdf",
            questions_json_path="vendor_risk_categories_detailed.json"
        )
        
        print("Step 3: Validating vendor assessment...")
        # Validate the assessment
        validation_summary = validator_agent.validate_vendor_assessment(
            questions_json_path="vendor_risk_categories_detailed.json",
            expected_answers=expected_answers,
            vendor_pdf_answers=vendor_pdf_answers
        )
        
        print("\\n" + "="*80)
        print("VENDOR RISK ASSESSMENT VALIDATION REPORT")
        print("="*80)
        print(validation_summary)
        
    except Exception as e:
        print(f"Error in validation workflow: {e}")


if __name__ == "__main__":
    # Set your Groq API key here or in environment variable
    # os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
    
    example_usage()