import os
import json
from typing import List
from enum import Enum
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from dotenv import load_dotenv

load_dotenv()


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


# FIX 1: Create a wrapper model for the list of answers
class PDFAssessmentResponse(BaseModel):
    """Response model that wraps a list of answers"""
    answers: List[PDFAssessmentAnswer] = Field(..., description="List of question-answer pairs")


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
            model_name="llama-3.1-8b-instant",
            temperature=0.1,  # Very low temperature for factual extraction
            max_tokens=8000
        )
        
        # FIX 2: Use the wrapper model instead of List[PDFAssessmentAnswer]
        self.output_parser = PydanticOutputParser(pydantic_object=PDFAssessmentResponse)
        
        # Create the prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create the chain
        self.chain = self._create_chain()
    
    def _read_pdf_content(self, pdf_path: str) -> str:
        """
        Read content from PDF file using LangChain community document loaders
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content from PDF
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"Attempting to read PDF using LangChain loaders: {pdf_path}")
        
        # Method 1: Try PyPDFLoader first
        try:
            print("Trying PyPDFLoader...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if documents:
                text_content = ""
                for i, doc in enumerate(documents):
                    page_content = doc.page_content.strip()
                    if page_content:
                        text_content += f"\n=== PAGE {i + 1} ===\n{page_content}\n"
                
                if text_content.strip():
                    print(f"PyPDFLoader: Successfully extracted {len(text_content)} characters from {len(documents)} pages")
                    return text_content
                else:
                    print("PyPDFLoader: No text content found in documents")
            else:
                print("PyPDFLoader: No documents loaded")
                
        except Exception as e:
            print(f"PyPDFLoader failed: {e}")
        
        # Method 2: Try UnstructuredPDFLoader as fallback
        try:
            print("Trying UnstructuredPDFLoader as fallback...")
            loader = UnstructuredPDFLoader(pdf_path)
            documents = loader.load()
            
            if documents:
                text_content = ""
                for i, doc in enumerate(documents):
                    page_content = doc.page_content.strip()
                    if page_content:
                        text_content += f"\n=== SECTION {i + 1} ===\n{page_content}\n"
                
                if text_content.strip():
                    print(f"UnstructuredPDFLoader: Successfully extracted {len(text_content)} characters")
                    return text_content
                else:
                    print("UnstructuredPDFLoader: No text content found")
            else:
                print("UnstructuredPDFLoader: No documents loaded")
                
        except Exception as e:
            print(f"UnstructuredPDFLoader failed: {e}")
        
        # Method 3: Fallback to basic file reading (in case it's a text file)
        try:
            print("Attempting to read as text file...")
            with open(pdf_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if content.strip():
                    print(f"Text file reading: Successfully extracted {len(content)} characters")
                    return f"\n=== DOCUMENT CONTENT ===\n{content}\n"
        except Exception as e:
            print(f"Text file reading failed: {e}")
        
        raise Exception("Unable to extract any text content from the file using any method. The file may be image-based, corrupted, or empty.")
    
    def _process_pdf_content(self, pdf_content: str) -> str:
        """
        Process and clean PDF content for better analysis
        
        Args:
            pdf_content: Raw text from PDF
            
        Returns:
            Processed PDF content
        """
        if not pdf_content or len(pdf_content.strip()) == 0:
            raise Exception("PDF content is empty after extraction")
        
        print(f"Processing PDF content: {len(pdf_content)} characters")
        
        # Clean up the text while preserving structure
        processed_content = pdf_content.replace('\n\n\n', '\n\n')  # Remove excessive newlines
        processed_content = processed_content.replace('\t', ' ')  # Replace tabs with spaces
        
        # Ensure we have substantial content
        if len(processed_content.strip()) < 100:
            raise Exception(f"PDF content too short after processing: {len(processed_content)} characters")
        
        # Truncate if too long (keep within LLM context limits)
        max_length = 12000  # Increased limit for better analysis
        if len(processed_content) > max_length:
            processed_content = processed_content[:max_length] + "\n\n[CONTENT TRUNCATED - PROCESSING FIRST PORTION OF PDF]"
            print(f"Content truncated to {max_length} characters")
        
        print(f"Final processed content: {len(processed_content)} characters")
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
You are an expert document analyst specializing in extracting specific information from vendor security documents.

CRITICAL INSTRUCTIONS:
1. CAREFULLY analyze the provided PDF content
2. ONLY use information explicitly stated in the PDF content
3. If information is not found in the PDF, answer "None"
4. Be precise and extract exact information when available
5. Look for keywords, phrases, and context that match the questions

PDF CONTENT TO ANALYZE:
{pdf_content}

QUESTIONS TO ANSWER:
{questions_json}

ANALYSIS GUIDELINES:

For SINGLE_CHOICE questions:
- Answer "Yes" ONLY if explicitly confirmed in the PDF (look for phrases like "we do", "we have", "we maintain", "yes", "implemented", "established")
- Answer "No" if explicitly denied in the PDF (look for phrases like "we do not", "no", "not implemented", "not established")
- Answer "None" if no relevant information is found in the PDF

For TEXT questions:
- Extract and summarize relevant information from the PDF using exact phrases when possible
- Combine related information from different parts of the document
- Answer "None" if no relevant information is found
- Use quotation marks for direct quotes from the document

IMPORTANT: Look carefully for information that might be phrased differently than the question but still answers it.

Examples of what to look for:
- Question about "ISO 27001" - look for "ISO 27001", "ISO certification", "information security management system"
- Question about "multi-factor authentication" - look for "MFA", "two-factor", "multi-factor", "authentication"
- Question about "backup procedures" - look for "backup", "recovery", "disaster recovery", "data protection"

{format_instructions}

Analyze the PDF content and provide answers in the specified JSON format:
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
        Assess PDF content against provided questions using LangChain document loaders
        
        Args:
            pdf_file_path: Path to the vendor PDF file
            questions_json_path: Path to JSON file with assessment questions
            
        Returns:
            List of PDFAssessmentAnswer with question-answer pairs
        """
        try:
            print("="*80)
            print("STARTING PDF ASSESSMENT WITH LANGCHAIN LOADERS")
            print("="*80)
            
            # Read PDF content using LangChain loaders
            pdf_content = self._read_pdf_content(pdf_file_path)
            pdf_content = self._process_pdf_content(pdf_content)
            
            # Load questions
            questions = self._load_questions_from_json(questions_json_path)
            print(f"Loaded {len(questions)} questions from JSON file")
            
            # FIX 3: Process questions in smaller batches and handle responses properly
            batch_size = 5  # Smaller batches for better reliability
            all_answers = []
            
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]
                print(f"\nProcessing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
                print(f"Questions {i+1} to {min(i+batch_size, len(questions))}")
                
                # Convert questions to JSON format for the prompt
                questions_data = []
                for q in batch_questions:
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
                
                try:
                    # Process through the chain
                    batch_response = self.chain.invoke(input_data)
                    
                    # FIX 4: Properly handle the response structure
                    if isinstance(batch_response, PDFAssessmentResponse) and batch_response.answers:
                        all_answers.extend(batch_response.answers)
                        print(f"✅ Successfully processed {len(batch_response.answers)} answers")
                    else:
                        # Fallback: create None answers for this batch
                        print("⚠️ No answers returned, creating None responses")
                        for q in batch_questions:
                            all_answers.append(PDFAssessmentAnswer(question=q.question, answer="None"))
                            
                except Exception as batch_error:
                    print(f"❌ Error processing batch: {batch_error}")
                    # Create None answers for failed batch
                    for q in batch_questions:
                        all_answers.append(PDFAssessmentAnswer(question=q.question, answer="None"))
            
            print(f"\n✅ COMPLETED PROCESSING {len(all_answers)} TOTAL QUESTIONS")
            return all_answers
            
        except Exception as e:
            print(f"❌ Critical error in PDF assessment: {e}")
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
            
            # FIX 5: Return the answers properly
            if isinstance(response, PDFAssessmentResponse):
                return response.answers
            else:
                return response
            
        except Exception as e:
            raise Exception(f"Error assessing PDF: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Set your Groq API key here or in environment variable
    # os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
    
    agent = PDFVendorAssessmentAgent()
    
    try:
        print("=" * 80)
        print("PDF VENDOR ASSESSMENT - LANGCHAIN DOCUMENT LOADERS")
        print("=" * 80)
        
        response = agent.assess_pdf_with_questions(
            pdf_file_path="secure.pdf",
            questions_json_path="vendor_risk_questions.json"
        )
        
        print(f"\n✅ SUCCESSFULLY PROCESSED {len(response)} QUESTIONS")
        print("=" * 80)
        
        # Categorize and display results
        categories = {}
        for answer in response:
            # Simple categorization based on keywords
            category = "General"
            question_lower = answer.question.lower()
            
            if any(keyword in question_lower for keyword in ["iso 27001", "soc 2", "certification", "penetration", "multi-factor", "cybersecurity", "incident response"]):
                category = "Information Security & Cybersecurity"
            elif any(keyword in question_lower for keyword in ["gdpr", "ccpa", "data protection", "dpo", "privacy", "dpia"]):
                category = "Data Privacy & Protection"
            elif any(keyword in question_lower for keyword in ["bankruptcy", "financial", "insurance", "business continuity", "disaster", "rto", "rpo"]):
                category = "Financial Stability & Business Continuity"
            elif any(keyword in question_lower for keyword in ["sla", "24/7", "support", "change management", "escalation", "monitoring"]):
                category = "Operational Risk & Service Delivery"
            elif any(keyword in question_lower for keyword in ["compliance", "regulatory", "audit", "certifications"]):
                category = "Compliance & Regulatory Adherence"
            elif any(keyword in question_lower for keyword in ["litigation", "contract", "liability", "intellectual property"]):
                category = "Legal & Contractual Considerations"
            elif any(keyword in question_lower for keyword in ["third-party", "subcontractor", "vendor", "due diligence"]):
                category = "Third-Party Management"
            elif any(keyword in question_lower for keyword in ["environmental", "sustainability", "diversity", "ethical"]):
                category = "Environmental & Social Governance"
            elif any(keyword in question_lower for keyword in ["cloud", "infrastructure", "aws", "azure", "backup", "disaster recovery"]):
                category = "Technology Infrastructure"
            elif any(keyword in question_lower for keyword in ["background", "employee", "training", "hr", "personnel", "byod"]):
                category = "Human Resources & Personnel Security"
            
            if category not in categories:
                categories[category] = []
            categories[category].append(answer)
        
        # Display results by category
        answered_count = 0
        for category, answers in categories.items():
            print(f"\n📋 {category.upper()}")
            print("-" * 60)
            for i, answer in enumerate(answers, 1):
                print(f"{i:2d}. Question: {answer.question}")
                print(f"    Answer: {answer.answer}")
                if answer.answer != "None":
                    answered_count += 1
                print()
        
        # Summary statistics
        total_questions = len(response)
        none_answers = total_questions - answered_count
        success_rate = (answered_count / total_questions) * 100 if total_questions > 0 else 0
        
        print("=" * 80)
        print("📊 SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total Questions Processed: {total_questions}")
        print(f"Questions with Answers: {answered_count}")
        print(f"Questions with 'None' (not found in PDF): {none_answers}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("=" * 80)
        
        if success_rate == 0:
            print("\n⚠️ WARNING: 0% success rate indicates potential issues:")
            print("1. PDF may be image-based or corrupted")
            print("2. PDF content extraction may have failed") 
            print("3. PDF content may not match the questions asked")
            print("4. LLM prompt may need adjustment")
            
    except Exception as e:
        print(f"❌ Error during PDF assessment: {e}")
        import traceback
        traceback.print_exc()