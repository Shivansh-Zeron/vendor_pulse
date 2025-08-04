import os
import json
import PyPDF2
import fitz  # PyMuPDF
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough


class QuestionType(str, Enum):
    """Types of assessment questions"""
    SINGLE_CHOICE = "single_choice"
    TEXT = "text"


class QuestionInput(BaseModel):
    """Input model for individual question"""
    question: str = Field(..., description="The question to be answered")
    question_type: QuestionType = Field(..., description="Type of question (single_choice or text)")
    question_weight: float = Field(..., ge=0, le=1, description="Weight/importance of the question")
    category: str = Field(..., description="Category the question belongs to")


class PDFAssessmentAnswer(BaseModel):
    """Response model for PDF-based assessment answers"""
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Answer based on PDF content or 'None' if not found")





class PDFVendorAssessmentAgent:
    """PDF-based vendor assessment agent that answers questions strictly from PDF content"""
    
    def __init__(self, groq_api_key: str = None):
        """
        Initialize the PDF assessment agent
        
        Args:
            groq_api_key: Groq API key (if not provided, will look for GROQ_API_KEY env var)
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly.")
        
        # Initialize the Groq LLM
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.1,  # Very low temperature for factual extraction
            max_tokens=8000
        )
        
        # Set up the output parser
        self.output_parser = PydanticOutputParser(pydantic_object=List[PDFAssessmentAnswer])
        
        # Create the prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create the chain
        self.chain = self._create_chain()
    
    def _read_pdf_content(self, pdf_path: str) -> str:
        """
        Read content from PDF file using multiple methods
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content from PDF
        """
        if not pdf_path or not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Method 1: Try PyMuPDF (fitz) first - generally more reliable
            try:
                doc = fitz.open(pdf_path)
                text_content = ""
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    # Add page reference for better tracking
                    text_content += f"\\n[PAGE {page_num + 1}]\\n{page_text}\\n"
                doc.close()
                
                if text_content.strip():
                    return text_content
            except Exception as e:
                print(f"PyMuPDF failed: {e}")
            
            # Method 2: Try PyPDF2 as fallback
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = ""
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        text_content += f"\\n[PAGE {page_num + 1}]\\n{page_text}\\n"
                    
                    if text_content.strip():
                        return text_content
            except Exception as e:
                print(f"PyPDF2 failed: {e}")
            
            raise Exception("Unable to extract text from PDF file. The file may be image-based or corrupted.")
            
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")
    
    def _process_pdf_content(self, pdf_content: str) -> str:
        """
        Process and clean PDF content for better analysis
        
        Args:
            pdf_content: Raw text from PDF
            
        Returns:
            Processed PDF content
        """
        # Clean up the text while preserving structure
        processed_content = pdf_content.replace('\\n\\n\\n', '\\n\\n')  # Remove excessive newlines
        
        # Truncate if too long (keep within LLM context limits)
        max_length = 15000  # Adjust based on your needs and context window
        if len(processed_content) > max_length:
            processed_content = processed_content[:max_length] + "\\n\\n[CONTENT TRUNCATED - PROCESSING FIRST PORTION OF PDF]"
        
        return processed_content
    
    def _load_questions_from_json(self, json_file_path: str) -> List[QuestionInput]:
        """
        Load questions from JSON file
        
        Args:
            json_file_path: Path to the JSON file containing questions
            
        Returns:
            List of QuestionInput objects
        """
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            questions = []
            for category, category_questions in data.items():
                for q in category_questions:
                    question_input = QuestionInput(
                        question=q["question"],
                        question_type=QuestionType(q["question_type"]),
                        question_weight=q["question_weight"],
                        category=category
                    )
                    questions.append(question_input)
            
            return questions
            
        except Exception as e:
            raise Exception(f"Error loading questions from JSON: {str(e)}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for PDF-based assessment"""
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert document analyst. Your task is to analyze the provided PDF content and answer specific questions based STRICTLY on the information contained within the document.

CRITICAL INSTRUCTIONS:
1. **ONLY use information explicitly stated in the PDF content**
2. **DO NOT use your general knowledge or make assumptions**
3. **If information is not found in the PDF, answer "None"**
4. **Be precise and factual**

PDF CONTENT TO ANALYZE:
{pdf_content}

QUESTIONS TO ANSWER:
{questions_json}

ANALYSIS GUIDELINES:

For SINGLE_CHOICE questions:
- Answer "Yes" only if explicitly confirmed in the PDF
- Answer "No" if explicitly denied in the PDF  
- Answer "None" if no information is found in the PDF

For TEXT questions:
- Extract and summarize relevant information from the PDF
- Answer "None" if no relevant information is found

Remember: Stick strictly to what is documented in the provided PDF content.

{format_instructions}

Analyze the PDF content and provide answers:
""")
        
        return prompt
    
    def _create_chain(self):
        """Create the LangChain processing chain"""
        
        chain = (
            {
                "pdf_content": RunnablePassthrough(),
                "questions_json": RunnablePassthrough(),
                "format_instructions": lambda _: self.output_parser.get_format_instructions()
            }
            | self.prompt_template
            | self.llm
            | self.output_parser
        )
        
        return chain
    
    def assess_pdf_with_questions(self, 
                                 pdf_file_path: str, 
                                 questions_json_path: str) -> List[PDFAssessmentAnswer]:
        """
        Assess PDF content against provided questions
        
        Args:
            pdf_file_path: Path to the vendor PDF file
            questions_json_path: Path to JSON file with assessment questions
            
        Returns:
            List of PDFAssessmentAnswer with question-answer pairs
        """
        try:
            # Read PDF content
            pdf_content = self._read_pdf_content(pdf_file_path)
            pdf_content = self._process_pdf_content(pdf_content)
            
            # Load questions
            questions = self._load_questions_from_json(questions_json_path)
            
            # Convert questions to JSON format for the prompt
            questions_data = []
            for q in questions:
                questions_data.append({
                    "question": q.question,
                    "question_type": q.question_type.value,
                    "question_weight": q.question_weight,
                    "category": q.category
                })
            
            questions_json = json.dumps(questions_data, indent=2)
            
            # Prepare input data
            input_data = {
                "pdf_content": pdf_content,
                "questions_json": questions_json
            }
            
            # Process through the chain
            response = self.chain.invoke(input_data)
            
            return response
            
        except Exception as e:
            raise Exception(f"Error assessing PDF: {str(e)}")
    
    def assess_pdf_with_question_list(self, 
                                    pdf_file_path: str, 
                                    questions: List[QuestionInput]) -> List[PDFAssessmentAnswer]:
        """
        Assess PDF content against a list of questions
        
        Args:
            pdf_file_path: Path to the vendor PDF file
            questions: List of QuestionInput objects
            
        Returns:
            List of PDFAssessmentAnswer with question-answer pairs
        """
        try:
            # Read PDF content
            pdf_content = self._read_pdf_content(pdf_file_path)
            pdf_content = self._process_pdf_content(pdf_content)
            
            # Convert questions to JSON format for the prompt
            questions_data = []
            for q in questions:
                questions_data.append({
                    "question": q.question,
                    "question_type": q.question_type.value,
                    "question_weight": q.question_weight,
                    "category": q.category
                })
            
            questions_json = json.dumps(questions_data, indent=2)
            
            # Prepare input data
            input_data = {
                "pdf_content": pdf_content,
                "questions_json": questions_json
            }
            
            # Process through the chain
            response = self.chain.invoke(input_data)
            
            return response
            
        except Exception as e:
            raise Exception(f"Error assessing PDF: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Set your Groq API key here or in environment variable
    # os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
    
    agent = PDFVendorAssessmentAgent()
    
    response = agent.assess_pdf_with_questions(
        pdf_file_path="vendor_document.pdf",
        questions_json_path="vendor_risk_categories_detailed.json"
    )
    
    for answer in response:
        print(f"Question: {answer.question}")
        print(f"Answer: {answer.answer}")
        print("-" * 50)