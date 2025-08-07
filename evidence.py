# Legal Evidence Agent - LangChain Implementation
# Dependencies: pip install langchain langchain-groq langchain-community google-generativeai pydantic pdf2image pillow

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import base64
import io
from pathlib import Path

# Core dependencies
from pydantic import BaseModel, Field, validator
from PIL import Image
import pdf2image

# LangChain imports
from langchain.agents import create_openai_functions_agent, AgentExecutor, initialize_agent, AgentType
from langchain.tools import BaseTool, tool
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Groq
from langchain_groq import ChatGroq

# LangChain Document Loaders
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.document_loaders.base import BaseLoader

# Gemini for vision
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# Configuration and Models
# ========================================

class Config:
    """Configuration management for the Evidence Agent"""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Legal compliance settings
    AUDIT_LOG_ENABLED = True
    DATA_RETENTION_DAYS = 2555  # 7 years for legal documents
    
    # Model configurations
    GROQ_MODEL = "llama3-8b-8192"
    GEMINI_MODEL = "gemini-1.5-flash"
    
    # Document processing
    CHUNK_SIZE = 4000
    CHUNK_OVERLAP = 200

class AnalysisRequest(BaseModel):
    """Simple request model for document analysis"""
    question: str = Field(..., min_length=1, description="Question about the document")
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

# ========================================
# LangChain Document Processor
# ========================================

class LegalDocumentLoader:
    """Simple document loader for legal documents"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_pdf_document(self, file_path: str) -> List[Document]:
        """Load PDF using LangChain PyPDFLoader"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split documents into chunks for better processing
            split_docs = self.text_splitter.split_documents(documents)
            
            logger.info(f"Loaded {len(documents)} pages, split into {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading PDF with PyPDFLoader: {str(e)}")
            try:
                # Fallback to UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(file_path)
                documents = loader.load()
                split_docs = self.text_splitter.split_documents(documents)
                logger.info(f"Fallback loader: {len(documents)} pages, {len(split_docs)} chunks")
                return split_docs
            except Exception as e2:
                logger.error(f"Error with fallback loader: {str(e2)}")
                raise ValueError(f"Failed to load PDF: {str(e)}")

# ========================================
# Gemini Vision Analyzer
# ========================================

class GeminiVisionAnalyzer:
    """Gemini-based visual analysis component"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
    
    def convert_pdf_to_images(self, file_path: str) -> List[Image.Image]:
        """Convert PDF pages to images for visual analysis"""
        try:
            images = pdf2image.convert_from_path(file_path)
            logger.info(f"Converted PDF to {len(images)} images")
            return images
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise ValueError(f"Failed to convert PDF to images: {str(e)}")
    
    def analyze_visual_content(self, file_path: str, question: str) -> Dict[str, Any]:
        """Analyze visual content using Gemini Vision"""
        try:
            images = self.convert_pdf_to_images(file_path)
            visual_analyses = []
            
            for idx, image in enumerate(images):
                # Create analysis prompt for visual elements
                visual_prompt = f"""Analyze this legal document page for visual elements relevant to the question: "{question}"

                Focus on:
                - Tables, charts, diagrams, and structured data
                - Signatures, stamps, seals, and official markings
                - Document formatting, layout, and structure
                - Any visual elements not captured in text extraction
                - Legal document features (letterheads, official markings, watermarks)
                - Currency symbols, dates in special formats
                - Handwritten notes or annotations
                
                Provide specific observations about visual elements that relate to the question.
                If no relevant visual elements are found, state this clearly."""
                
                # Analyze with Gemini Vision
                try:
                    response = self.model.generate_content([visual_prompt, image])
                    analysis_text = response.text if hasattr(response, 'text') else str(response)
                except Exception as e:
                    analysis_text = f"Error analyzing page {idx + 1}: {str(e)}"
                
                visual_analyses.append({
                    'page_number': idx + 1,
                    'visual_analysis': analysis_text,
                    'image_size': image.size
                })
            
            return {
                'visual_analyses': visual_analyses,
                'total_pages_analyzed': len(images),
                'model_used': Config.GEMINI_MODEL
            }
            
        except Exception as e:
            logger.error(f"Error in Gemini vision analysis: {str(e)}")
            return {
                'visual_analyses': [{'page_number': 1, 'visual_analysis': f"Vision analysis failed: {str(e)}", 'image_size': (0, 0)}],
                'total_pages_analyzed': 0,
                'model_used': Config.GEMINI_MODEL
            }

# ========================================
# LangChain Tools
# ========================================

@tool
def analyze_document_comprehensive(question: str, documents: str) -> str:
    """Comprehensive analysis of legal documents including text analysis and compliance checking."""
    try:
        # Initialize LangChain Groq
        llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.GROQ_MODEL,
            temperature=0.1
        )
        
        # Create comprehensive analysis prompt
        prompt = PromptTemplate(
            input_variables=["question", "documents"],
            template="""You are a legal evidence analysis expert. Perform a comprehensive analysis of the provided document text that includes both answering the specific question and checking legal compliance.

ANALYSIS GUIDELINES:
- Provide accurate, evidence-based responses
- Quote relevant sections with specific references
- Identify any legal implications or compliance issues
- Assign confidence levels to your findings
- Note any missing or unclear information
- Maintain objectivity and precision
- Focus on legally relevant information

COMPLIANCE CHECK AREAS:
1. Document authenticity indicators
2. Required legal elements and clauses
3. Compliance with relevant regulations
4. Data privacy considerations
5. Evidence admissibility factors
6. Completeness of legal requirements
7. Potential legal risks or issues

Document Content:
{documents}

Question: {question}

Provide a comprehensive analysis with:

1. DIRECT ANSWER TO QUESTION:
   - Clear, specific answer to the question asked
   - Supporting evidence from the document with quotes
   - Page/section references where possible

2. LEGAL COMPLIANCE ANALYSIS:
   - Document authenticity and completeness assessment
   - Required legal elements present/missing
   - Regulatory compliance observations
   - Privacy and data protection considerations
   - Evidence admissibility factors

3. LEGAL IMPLICATIONS:
   - Key legal considerations arising from the content
   - Potential risks or issues identified
   - Recommendations for further review if needed

4. CONFIDENCE AND LIMITATIONS:
   - Confidence level in the findings (High/Medium/Low)
   - Any limitations in the analysis
   - Missing information that might affect conclusions

Format your response clearly with these sections for easy readability."""
        )
        
        # Create and run chain
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(question=question, documents=documents)
        
        return result
        
    except Exception as e:
        return f"Error in comprehensive document analysis: {str(e)}"

@tool
def synthesize_multimodal_analysis(text_analysis: str, visual_analysis: str, question: str) -> str:
    """Synthesize text and visual analyses into a comprehensive legal response."""
    try:
        llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.GROQ_MODEL,
            temperature=0.1
        )
        
        prompt = PromptTemplate(
            input_variables=["text_analysis", "visual_analysis", "question"],
            template="""Synthesize the following analyses to provide a comprehensive legal answer.

Original Question: {question}

Text Analysis Results:
{text_analysis}

Visual Analysis Results:
{visual_analysis}

Provide a synthesized response that includes:
1. Comprehensive answer combining both analyses
2. Confidence score (High/Medium/Low) based on evidence quality
3. Key sources and evidence found
4. Legal compliance observations
5. Recommendations for further action if needed

Format as a professional legal analysis report."""
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(
            text_analysis=text_analysis,
            visual_analysis=visual_analysis,
            question=question
        )
        
        return result
        
    except Exception as e:
        return f"Error in synthesis: {str(e)}"

# ========================================
# Main Evidence Agent
# ========================================

class LegalEvidenceAgent:
    """Simple evidence analysis agent"""
    
    def __init__(self, groq_api_key: str, gemini_api_key: str):
        # Validate API keys
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        # Initialize components
        self.doc_loader = LegalDocumentLoader()
        self.vision_analyzer = GeminiVisionAnalyzer(gemini_api_key)
        
        # Initialize LangChain Groq
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=Config.GROQ_MODEL,
            temperature=0.1
        )
        
        # Initialize tools
        self.tools = [
            analyze_document_comprehensive,
            synthesize_multimodal_analysis
        ]
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def analyze_document(self, file_path: str, question: str) -> str:
        """Analyze document and return answer as string"""
        try:
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            if not file_path.lower().endswith('.pdf'):
                raise ValueError("Only PDF files are supported")
            
            # Load document using LangChain
            documents = self.doc_loader.load_pdf_document(file_path)
            
            # Combine all document content
            full_text = "\n\n".join([doc.page_content for doc in documents])
            
            # Perform comprehensive text analysis (includes compliance check)
            print("🔍 Performing comprehensive document analysis...")
            text_analysis = analyze_document_comprehensive.run(
                question=question,
                documents=full_text
            )
            
            # Perform visual analysis
            print("👁️ Performing visual analysis...")
            visual_result = self.vision_analyzer.analyze_visual_content(file_path, question)
            visual_analysis = str(visual_result['visual_analyses'])
            
            # Synthesize all analyses
            print("🔗 Synthesizing analyses...")
            final_answer = synthesize_multimodal_analysis.run(
                text_analysis=text_analysis,
                visual_analysis=visual_analysis,
                question=question
            )
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error in document analysis: {str(e)}")
            return f"Analysis failed: {str(e)}"

# ========================================
# Usage Functions
# ========================================

def main():
    """Simple main function for basic usage"""
    print("🚀 Legal Evidence Agent")
    print("=" * 40)
    
    # Get file path and question
    file_path = input("PDF file path: ").strip()
    question = input("Your question: ").strip()
    
    try:
        print("\n🔄 Analyzing...")
        answer = analyze_single_document(file_path, question)
        print(f"\n📊 Answer:\n{answer}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def analyze_single_document(file_path: str, question: str) -> str:
    """Analyze a single document and return the answer"""
    agent = LegalEvidenceAgent(
        groq_api_key=Config.GROQ_API_KEY,
        gemini_api_key=Config.GEMINI_API_KEY
    )
    return agent.analyze_document(file_path, question)

# ========================================
# Example Usage
# ========================================

if __name__ == "__main__":
    # Check if API keys are set
    if not Config.GROQ_API_KEY or not Config.GEMINI_API_KEY:
        print("❌ Please set your API keys:")
        print("export GROQ_API_KEY='your_groq_api_key'")
        print("export GEMINI_API_KEY='your_gemini_api_key'")
    else:
        main()