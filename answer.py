import os
import json
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Import from question.py
from question import Question, QuestionCategory, VendorAssessmentQuestionnaire, QuestionType

load_dotenv()

class VendorAnswer(BaseModel):
    """Response model for vendor answers"""
    question_id: str = Field(..., description="Unique identifier for the question")
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Vendor's expected response to the question")
    question_category: str = Field(..., description="Category the question belongs to")
    evidence_required: bool = Field(..., description="Whether document evidence is required")
    suggested_evidence: str = Field(..., description="Suggested evidence to provide")

class VendorAssessmentResponse(BaseModel):
    """Complete vendor assessment response"""
    answers: List[VendorAnswer] = Field(..., description="List of all question-answer pairs")

class VendorAnswerAgent:
    """Third-party vendor risk assessment answer agent"""
    
    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly.")
        
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=8000
        )
        
        self.output_parser = PydanticOutputParser(pydantic_object=VendorAssessmentResponse)
        self.prompt_template = self._create_prompt_template()
        self.chain = self._create_chain()
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_template("""
You are an expert third-party vendor representative with over 15 years of experience in cybersecurity, 
compliance, and risk management. You work for a professional technology services organization that provides 
services to enterprise clients and maintains high security and compliance standards.

Your expertise includes:
- Information Security Management (ISO 27001, SOC 2, NIST Framework)
- Compliance and Regulatory Requirements (GDPR, HIPAA, PCI-DSS, SOX)
- Risk Management and Business Continuity Planning
- Access Control and Identity Management
- Data Protection and Privacy Management
- Cloud Security and Infrastructure Management
- Incident Response and Security Operations
- Vendor Risk Management and Third-Party Assessments
- Security Architecture and Development Practices

INSTRUCTIONS:
You are responding to a vendor risk assessment questionnaire from a potential client. Provide the expected 
professional answers that demonstrate your organization's strong security posture and compliance practices.

RESPONSE GUIDELINES:

For SINGLE_CHOICE questions:
- Provide "Yes" for security best practices and compliance requirements
- Give brief explanatory context (1-2 sentences)
- Mention relevant certifications or standards when applicable

For TEXT questions:
- Provide detailed, comprehensive responses (3-5 sentences)
- Include specific methodologies, technologies, and frameworks
- Reference industry standards and best practices
- Mention frequency of reviews, testing, and updates
- Provide examples of tools, procedures, or policies

For questions requiring evidence:
- Suggest specific documents, certificates, or artifacts to provide
- Include policy names, audit reports, compliance certificates
- Mention specific standards or frameworks being followed

QUESTIONS TO ANSWER:
{questions_json}

Return a JSON object with this exact structure:

{{
  "answers": [
    {{
      "question_id": "Q1",
      "question": "Does the vendor have a documented framework compliance program?",
      "answer": "Yes. Our organization maintains a comprehensive framework compliance program aligned with NIST Cybersecurity Framework and ISO 27001 standards. We have dedicated compliance officers and regular audits to ensure adherence.",
      "question_category": "Framework Compliance & Governance",
      "evidence_required": true,
      "suggested_evidence": "ISO 27001 certification, NIST CSF assessment report, compliance program documentation, audit reports"
    }}
  ]
}}

CRITICAL: Return ONLY the JSON object with the answers array, no additional text or explanations.

{format_instructions}
""")
        
        return prompt
    
    def _create_chain(self):
        chain = (
            {
                "questions_json": RunnablePassthrough(),
                "format_instructions": lambda _: self.output_parser.get_format_instructions()
            }
            | self.prompt_template
            | self.llm
            | self.output_parser
        )
        
        return chain
    
    def generate_answers_from_questionnaire(self, questionnaire: VendorAssessmentQuestionnaire) -> VendorAssessmentResponse:
        """
        Generate vendor answers from a VendorAssessmentQuestionnaire
        
        Args:
            questionnaire: VendorAssessmentQuestionnaire object from question.py
            
        Returns:
            VendorAssessmentResponse with all answers and assessments
        """
        try:
            # Convert questionnaire questions to JSON format for the prompt
            questions_data = []
            for category in questionnaire.categories:
                for question in category.questions:
                    questions_data.append({
                        "question_id": question.question_id,
                        "question": question.question_text,
                        "question_type": question.question_type.value,
                        "question_weight": question.question_weight,
                        "category": category.category_name,
                        "evidence_required": question.evidence_required
                    })
            
            questions_json = json.dumps(questions_data, indent=2)
            
            # Prepare input data
            input_data = {
                "questions_json": questions_json
            }
            
            # Process through the chain
            response = self.chain.invoke(input_data)
            
            return response
            
        except Exception as e:
            raise Exception(f"Error generating vendor responses: {str(e)}")
    
    def _load_questions_from_json(self, json_file_path: str) -> List[Question]:
        """
        Load questions from JSON file (legacy method)
        
        Args:
            json_file_path: Path to the JSON file containing questions
            
        Returns:
            List of Question objects
        """
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            questions = []
            for category, category_questions in data.items():
                for q in category_questions:
                    question = Question(
                        question_id=q.get("question_id", "Q1"),
                        question_text=q["question"],
                        question_type=QuestionType(q["question_type"]),
                        question_weight=q["question_weight"],
                        rationale=q.get("rationale", "Risk assessment question"),
                        expected_answer=q.get("expected_answer", "Yes"),
                        red_flags=q.get("red_flags", ["No"]),
                        evidence_required=q.get("evidence_required", False)
                    )
                    questions.append(question)
            
            return questions
            
        except Exception as e:
            raise Exception(f"Error loading questions from JSON: {str(e)}")
    
    def answer_questions_from_json(self, json_file_path: str) -> VendorAssessmentResponse:
        """
        Generate vendor responses from JSON file containing questions (legacy method)
        
        Args:
            json_file_path: Path to JSON file with questions
            
        Returns:
            VendorAssessmentResponse with all answers and assessments
        """
        try:
            questions = self._load_questions_from_json(json_file_path)
            
            # Convert questions to JSON format for the prompt
            questions_data = []
            for q in questions:
                questions_data.append({
                    "question_id": q.question_id,
                    "question": q.question_text,
                    "question_type": q.question_type.value,
                    "question_weight": q.question_weight,
                    "category": "General",
                    "evidence_required": q.evidence_required
                })
            
            questions_json = json.dumps(questions_data, indent=2)
            
            # Prepare input data
            input_data = {
                "questions_json": questions_json
            }
            
            # Process through the chain
            response = self.chain.invoke(input_data)
            
            return response
            
        except Exception as e:
            raise Exception(f"Error generating vendor responses: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Import the questionnaire generator
    from question import VendorQuestionnaireAgent, VendorInput, Industry, VendorSize
    
    # Create a sample vendor input
    vendor_input = VendorInput(
        template_name="Cloud Service Provider Assessment",
        industry=Industry.TECHNOLOGY,
        vendor_size=VendorSize.LARGE,
        vendor_description="Cloud infrastructure provider offering IaaS, PaaS, and SaaS solutions including data storage, computing resources, AI/ML services, and enterprise applications. Serves Fortune 500 companies with global data centers."
    )
    
    # Generate questionnaire
    questionnaire_agent = VendorQuestionnaireAgent()
    questionnaire = questionnaire_agent.generate_questionnaire(vendor_input)
    
    print("Generated Questionnaire:")
    print(f"Assessment Overview: {questionnaire.assessment_overview}")
    print(f"Total Questions: {questionnaire.total_questions}")
    print(f"Categories: {len(questionnaire.categories)}")
    print("\n" + "="*80 + "\n")
    
    # Generate answers
    answer_agent = VendorAnswerAgent()
    response = answer_agent.generate_answers_from_questionnaire(questionnaire)
    
    print("Generated Vendor Answers:")
    print("="*80)
    
    for answer in response.answers:
        print(f"Question ID: {answer.question_id}")
        print(f"Category: {answer.question_category}")
        print(f"Question: {answer.question}")
        print(f"Answer: {answer.answer}")
        print(f"Evidence Required: {'Yes' if answer.evidence_required else 'No'}")
        if answer.evidence_required:
            print(f"Suggested Evidence: {answer.suggested_evidence}")
        print("-" * 50)