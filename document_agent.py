import streamlit as st
import os
import tempfile
import json
import time
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Document processing
import PyPDF2
from docx import Document
import google.generativeai as genai

# Vector storage and embeddings
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Configure Gemini with correct API key and model
GEMINI_API_KEY = "ENTER_YOUR_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []
        self.document_metadata = []
        
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF files with improved error handling"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"Page {page_num + 1}:\n{page_text}\n\n"
                
                if not text.strip():
                    return "No extractable text found in PDF. This might be a scanned document or image-based PDF."
                
                return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            return text if text.strip() else "No text content found in DOCX file."
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"
    
    def extract_text_from_txt(self, file_path):
        """Extract text from TXT files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return content if content.strip() else "File appears to be empty."
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                    return content if content.strip() else "File appears to be empty."
            except Exception as e:
                return f"Error reading TXT file: {str(e)}"
        except Exception as e:
            return f"Error reading TXT file: {str(e)}"

class DocumentAnalyzer:
    def __init__(self):
        try:
            # Correct updated Gemini model for v1beta
            self.gemini_model = genai.GenerativeModel(
                model_name="models/gemini-2.5-flash"
            )
        except Exception as e:
            st.error(f"Error initializing Gemini model: {str(e)}")
            self.gemini_model = None
    
    def generate_summary(self, text, max_length=500):
        if self.gemini_model is None:
            return "Gemini model not available. Please check your API key and model configuration."
        
        try:
            if text.startswith("Error") or len(text.strip()) < 50:
                return "Unable to generate summary: " + text

            prompt = f"""
            Summarize the following document in around {max_length} words.
            Focus on key points, findings, and main ideas.

            {text[:12000]}
            """

            response = self.gemini_model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def extract_key_insights(self, text):
        if self.gemini_model is None:
            return "Gemini model not available. Please check your API key and model configuration."

        try:
            if text.startswith("Error") or len(text.strip()) < 50:
                return "Unable to extract insights: " + text

            prompt = f"""
            Extract the most important insights from the text.
            Provide a bullet list of:
            - Key findings
            - Patterns
            - Conclusions

            Text:
            {text[:10000]}
            """

            response = self.gemini_model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            return f"Error extracting insights: {str(e)}"
    
    def answer_question(self, question, context):
        if self.gemini_model is None:
            return "Gemini model not available. Please check your API key and model configuration."

        try:
            prompt = f"""
            Using the document context below, answer the question:

            Question: {question}

            Context:
            {context[:8000]}
            """

            response = self.gemini_model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    
    def extract_key_insights(self, text):
        """Extract key insights using Gemini"""
        if self.gemini_model is None:
            return "Gemini model not available. Please check your API key and model configuration."
        
        try:
            # Check if text is an error message or too short
            if text.startswith("Error") or text.startswith("No ") or len(text.strip()) < 50:
                return "Unable to extract insights: " + text
            
            prompt = f"""
            Analyze the following text and extract the most important insights, key findings, and notable information. 
            Format the response as a bulleted list with clear, concise points. Focus on:
            - Main concepts and ideas
            - Key findings or results
            - Important statistics or data
            - Conclusions or recommendations
            - Notable patterns or trends
            
            Text:
            {text[:10000]}
            
            Key Insights:
            """
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error extracting insights: {str(e)}"
    
    def answer_question(self, question, context):
        """Answer questions based on document context"""
        if self.gemini_model is None:
            return "Gemini model not available. Please check your API key and model configuration."
        
        try:
            prompt = f"""
            Based on the following document content, please provide a detailed answer to this question: {question}
            
            Document Context:
            {context[:8000]}
            
            Provide a comprehensive answer with relevant details from the document:
            """
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"

class KnowledgeBase:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.analyzer = DocumentAnalyzer()
        self.knowledge_data = {}
        self.setup_vector_store()
    
    def setup_vector_store(self):
        """Initialize FAISS vector store"""
        self.processor.index = faiss.IndexFlatL2(384)  # Dimension for all-MiniLM-L6-v2
    
    def add_document(self, file_path, filename):
        """Process and add document to knowledge base"""
        try:
            # Extract text based on file type
            if filename.lower().endswith('.pdf'):
                text = self.processor.extract_text_from_pdf(file_path)
            elif filename.lower().endswith('.docx'):
                text = self.processor.extract_text_from_docx(file_path)
            elif filename.lower().endswith('.txt'):
                text = self.processor.extract_text_from_txt(file_path)
            else:
                return False, "Unsupported file format"
            
            # Check if text extraction was successful
            if any(error_indicator in text.lower() for error_indicator in ['error', 'no extractable', 'no text', 'empty']):
                return False, f"Text extraction issue: {text}"
            
            # Check if text is too short
            if len(text.strip()) < 100:
                return False, f"Extracted text seems too short ({len(text)} characters). This might be a scanned PDF or protected document."
            
            # Generate embeddings
            embedding = self.processor.model.encode([text])
            
            # Add to vector store
            if self.processor.index.ntotal == 0:
                self.processor.index = faiss.IndexFlatL2(embedding.shape[1])
            
            self.processor.index.add(embedding)
            self.processor.documents.append(text)
            
            # Generate summary and insights
            summary = self.analyzer.generate_summary(text)
            insights = self.analyzer.extract_key_insights(text)
            
            # Store metadata
            doc_metadata = {
                'filename': filename,
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'summary': summary,
                'insights': insights,
                'text_length': len(text),
                'embedding_index': len(self.processor.documents) - 1,
                'file_type': filename.split('.')[-1].upper(),
                'processed_successfully': True
            }
            
            self.processor.document_metadata.append(doc_metadata)
            self.knowledge_data[filename] = doc_metadata
            
            return True, "Document processed successfully"
            
        except Exception as e:
            return False, f"Processing error: {str(e)}"
    
    def search_similar_documents(self, query, k=3):
        """Search for similar documents using semantic search"""
        if not self.processor.documents:
            return []
        
        try:
            query_embedding = self.processor.model.encode([query])
            distances, indices = self.processor.index.search(query_embedding, min(k, len(self.processor.documents)))
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.processor.document_metadata):
                    results.append({
                        'document': self.processor.document_metadata[idx],
                        'similarity_score': float(1 / (1 + distances[0][i])),
                        'content': self.processor.documents[idx][:500] + "..." if len(self.processor.documents[idx]) > 500 else self.processor.documents[idx]
                    })
            
            return results
        except Exception as e:
            return []
    
    def query_knowledge_base(self, question):
        """Query the knowledge base for answers"""
        similar_docs = self.search_similar_documents(question)
        
        if not similar_docs:
            return {
                'answer': "No relevant documents found to answer your question.",
                'source_document': "None",
                'confidence': 0.0,
                'relevant_docs': []
            }
        
        # Use the most relevant document as context
        context = similar_docs[0]['content']
        answer = self.analyzer.answer_question(question, context)
        
        return {
            'answer': answer,
            'source_document': similar_docs[0]['document']['filename'],
            'confidence': similar_docs[0]['similarity_score'],
            'relevant_docs': similar_docs
        }
    
    def to_dict(self):
        """Convert knowledge base to dictionary for session state"""
        return {
            'documents': self.processor.documents,
            'document_metadata': self.processor.document_metadata,
            'knowledge_data': self.knowledge_data
        }
    
    def from_dict(self, data):
        """Load knowledge base from dictionary"""
        self.processor.documents = data.get('documents', [])
        self.processor.document_metadata = data.get('document_metadata', [])
        self.knowledge_data = data.get('knowledge_data', {})
        
        # Rebuild FAISS index
        if self.processor.documents:
            embeddings = self.processor.model.encode(self.processor.documents)
            self.processor.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.processor.index.add(embeddings)

# Streamlit App
class DocumentAgentApp:
    def __init__(self):
        self.setup_page()
        self.initialize_session_state()
        self.kb = st.session_state.kb
    
    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Document Autonomous Agent",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(45deg, #1f77b4, #2e86ab);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2e86ab;
            margin-bottom: 1rem;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 0.5rem;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .info-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
        }
        .document-card {
            padding: 1.5rem;
            border-radius: 0.8rem;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #dee2e6;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(45deg, #1f77b4, #2e86ab);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'kb' not in st.session_state:
            st.session_state.kb = KnowledgeBase()
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
    
    def save_knowledge_base(self):
        """Save knowledge base to session state"""
        st.session_state.kb_data = self.kb.to_dict()
    
    def render_sidebar(self):
        """Render the sidebar with navigation and stats"""
        with st.sidebar:
            st.markdown("<h2 style='text-align: center; color: #1f77b4;'>📚 DocAgent Pro</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            # Navigation
            page = st.radio(
                "Navigation",
                ["🏠 Dashboard", "📤 Upload Documents", "🔍 Search & Query", "📊 Analytics", "🧠 Knowledge Base"],
                label_visibility="collapsed"
            )
            
            # Statistics
            st.markdown("---")
            st.markdown("### 📈 Statistics")
            total_docs = len(self.kb.processor.document_metadata)
            st.metric("Total Documents", total_docs)
            
            if total_docs > 0:
                total_text = sum(doc['text_length'] for doc in self.kb.processor.document_metadata)
                avg_length = total_text // total_docs
                st.metric("Avg Document Length", f"{avg_length:,} chars")
                
                successful_docs = sum(1 for doc in self.kb.processor.document_metadata if doc.get('processed_successfully', True))
                st.metric("Successfully Processed", successful_docs)
            
            st.markdown("---")
            st.markdown("### 🛠️ Actions")
            if st.button("🔄 Clear All Documents", use_container_width=True):
                st.session_state.kb = KnowledgeBase()
                st.session_state.processed_files = set()
                st.success("Knowledge base cleared!")
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 💡 Tips")
            st.info("""
            - Upload PDF, DOCX, or TXT files
            - For best results, use text-based PDFs
            - Scanned PDFs may not work well
            - Larger documents take longer to process
            """)
            
            return page
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.markdown("<h1 class='main-header'>🤖 Document Autonomous Agent</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Overview")
            total_docs = len(self.kb.processor.document_metadata)
            st.metric("Documents Processed", total_docs)
            
            if total_docs > 0:
                total_text = sum(doc['text_length'] for doc in self.kb.processor.document_metadata)
                st.metric("Total Text Processed", f"{total_text:,} chars")
        
        with col2:
            st.markdown("### 🎯 Features")
            features = [
                "✅ Automatic Summarization",
                "✅ Insight Extraction",
                "✅ Semantic Search",
                "✅ Q&A System",
                "✅ Continuous Learning",
                "✅ Multi-format Support",
                "✅ Real-time Analytics",
                "✅ Knowledge Base Explorer"
            ]
            for feature in features:
                st.write(feature)
        
        with col3:
            st.markdown("### 🚀 Quick Start")
            st.write("1. **Upload documents** in Upload section")
            st.write("2. **View summaries & insights** in Knowledge Base")
            st.write("3. **Ask questions** in Search & Query")
            st.write("4. **Explore analytics** in Analytics")
        
        st.markdown("---")
        
        # Recent documents
        if self.kb.processor.document_metadata:
            st.markdown("### 📄 Recent Documents")
            recent_docs = self.kb.processor.document_metadata[-5:]  # Last 5 documents
            
            for doc in reversed(recent_docs):
                with st.container():
                    st.markdown(f"""
                    <div class="document-card">
                        <h4>📋 {doc['filename']}</h4>
                        <p><strong>Uploaded:</strong> {doc['upload_time']}</p>
                        <p><strong>Summary Preview:</strong> {doc['summary'][:150]}...</p>
                        <p><strong>Text Length:</strong> {doc['text_length']:,} characters</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Quick actions for each document
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"View Details 👀", key=f"view_{doc['filename']}", use_container_width=True):
                            st.session_state.selected_doc = doc['filename']
                            st.switch_page("🧠 Knowledge Base")
                    with col2:
                        if st.button(f"Ask Questions ❓", key=f"ask_{doc['filename']}", use_container_width=True):
                            st.session_state.quick_question = f"Tell me about {doc['filename']}"
                            st.switch_page("🔍 Search & Query")
                    
                    st.markdown("---")
        else:
            st.info("""
            ## 📥 No documents uploaded yet!
            
            Go to **'Upload Documents'** to get started with your document analysis journey.
            
            Supported formats:
            - 📄 PDF (text-based)
            - 📝 DOCX (Word documents)  
            - 📃 TXT (Plain text files)
            """)
    
    def render_upload(self):
        """Render document upload interface"""
        st.markdown("<h2 class='sub-header'>📤 Upload Documents</h2>", unsafe_allow_html=True)
        
        # File upload section
        uploaded_files = st.file_uploader(
            "Choose documents to upload",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files. Maximum 200MB per file."
        )
        
        if uploaded_files:
            st.success(f"📁 Selected {len(uploaded_files)} file(s) for processing")
            
            # Show selected files
            for uploaded_file in uploaded_files:
                file_size = len(uploaded_file.getvalue()) / 1024
                st.write(f"📄 **{uploaded_file.name}** ({file_size:.1f} KB)")
            
            if st.button("🚀 Process Uploaded Documents", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                successful_uploads = 0
                failed_uploads = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Skip if already processed
                    if uploaded_file.name in st.session_state.processed_files:
                        st.warning(f"⚠️ {uploaded_file.name} was already processed. Skipping.")
                        continue
                    
                    # Update progress
                    progress = (i) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process document
                    success, message = self.kb.add_document(tmp_path, uploaded_file.name)
                    
                    # Clean up temporary file
                    os.unlink(tmp_path)
                    
                    # Add to processed files
                    if success:
                        st.session_state.processed_files.add(uploaded_file.name)
                    
                    # Show result
                    with results_container:
                        if success:
                            st.success(f"✅ **{uploaded_file.name}** - {message}")
                            successful_uploads += 1
                        else:
                            st.error(f"❌ **{uploaded_file.name}** - {message}")
                            failed_uploads.append((uploaded_file.name, message))
                    
                    # Small delay to show progress
                    time.sleep(0.5)
                
                # Final progress update
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                # Summary
                st.markdown("---")
                if successful_uploads > 0:
                    st.success(f"🎉 Successfully processed {successful_uploads} out of {len(uploaded_files)} documents!")
                    if successful_uploads > 0:
                        st.info("💡 Go to **'Knowledge Base'** to view summaries and insights, or **'Search & Query'** to ask questions about your documents.")
                
                if failed_uploads:
                    st.warning(f"⚠️ {len(failed_uploads)} documents failed to process:")
                    for failed_file, error_msg in failed_uploads:
                        st.write(f"• **{failed_file}**: {error_msg}")
                
                # Force a rerun to update the state
                st.rerun()
    
    def render_search(self):
        """Render search and query interface"""
        st.markdown("<h2 class='sub-header'>🔍 Search & Query</h2>", unsafe_allow_html=True)
        
        if not self.kb.processor.document_metadata:
            st.info("""
            ## 📚 No documents in knowledge base!
            
            Please upload some documents first to enable search and Q&A functionality.
            
            Go to **'Upload Documents'** to add documents to your knowledge base.
            """)
            return
        
        tab1, tab2 = st.tabs(["🤔 Ask Questions", "🔎 Semantic Search"])
        
        with tab1:
            st.markdown("### 💬 Ask Questions About Your Documents")
            
            # Quick question suggestions
            st.markdown("**💡 Quick Questions:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Main Topics", use_container_width=True):
                    st.session_state.quick_question = "What are the main topics discussed in the documents?"
            with col2:
                if st.button("Key Findings", use_container_width=True):
                    st.session_state.quick_question = "What are the key findings or results mentioned?"
            with col3:
                if st.button("Methodology", use_container_width=True):
                    st.session_state.quick_question = "What methodology or approach was used?"
            
            question = st.text_area(
                "Enter your question:",
                value=st.session_state.get('quick_question', ''),
                placeholder="e.g., What are the main concepts discussed in the documents? What are the key findings?",
                height=100,
                key="question_input"
            )
            
            if st.button("🎯 Get Answer", type="primary", use_container_width=True):
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("🔍 Searching through documents and generating answer..."):
                        result = self.kb.query_knowledge_base(question)
                    
                    st.markdown("---")
                    st.markdown("### 💡 Answer")
                    st.write(result['answer'])
                    
                    st.markdown("### 📋 Source Information")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Source Document", result['source_document'])
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.2%}")
                    with col3:
                        st.metric("Relevant Docs Found", len(result['relevant_docs']))
        
        with tab2:
            st.markdown("### 🔍 Search Similar Documents")
            search_query = st.text_input(
                "Enter search query:",
                placeholder="e.g., artificial intelligence, machine learning, data analysis"
            )
            
            if st.button("🔎 Search Documents", use_container_width=True):
                if not search_query.strip():
                    st.warning("Please enter a search query.")
                else:
                    with st.spinner("Searching for similar documents..."):
                        results = self.kb.search_similar_documents(search_query, k=5)
                    
                    if results:
                        st.markdown(f"### 📄 Found {len(results)} relevant documents")
                        
                        for i, result in enumerate(results):
                            with st.expander(
                                f"📋 {result['document']['filename']} "
                                f"(Similarity: {result['similarity_score']:.2%})", 
                                expanded=i==0
                            ):
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.write("**Document Info:**")
                                    st.write(f"- File: {result['document']['filename']}")
                                    st.write(f"- Type: {result['document']['file_type']}")
                                    st.write(f"- Length: {result['document']['text_length']:,} chars")
                                    st.write(f"- Uploaded: {result['document']['upload_time']}")
                                
                                with col2:
                                    st.write("**Summary:**")
                                    st.write(result['document']['summary'][:300] + "..." 
                                           if len(result['document']['summary']) > 300 
                                           else result['document']['summary'])
                                    
                                    st.write("**Relevant Content Preview:**")
                                    st.write(result['content'])
                    else:
                        st.info("No relevant documents found for your search query.")
    
    def render_analytics(self):
        """Render analytics dashboard"""
        st.markdown("<h2 class='sub-header'>📊 Document Analytics</h2>", unsafe_allow_html=True)
        
        if not self.kb.processor.document_metadata:
            st.info("""
            ## 📈 No analytics data available!
            
            Upload documents to see detailed analytics and insights about your document collection.
            """)
            return
        
        # Document statistics
        doc_data = []
        for meta in self.kb.processor.document_metadata:
            doc_data.append({
                'Filename': meta['filename'],
                'Text Length': meta['text_length'],
                'Upload Time': meta['upload_time'],
                'File Type': meta['file_type'],
                'Success': meta.get('processed_successfully', True)
            })
        
        df = pd.DataFrame(doc_data)
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", len(df))
        with col2:
            st.metric("Total Text", f"{df['Text Length'].sum():,} chars")
        with col3:
            st.metric("Average Length", f"{df['Text Length'].mean():.0f} chars")
        with col4:
            st.metric("File Types", df['File Type'].nunique())
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Document length distribution
            fig = px.histogram(df, x='Text Length', title='📏 Document Length Distribution',
                              color_discrete_sequence=['#1f77b4'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # File type distribution
            file_type_counts = df['File Type'].value_counts()
            fig = px.pie(values=file_type_counts.values, 
                        names=file_type_counts.index, 
                        title='📊 File Type Distribution',
                        color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        
        # Document timeline
        if len(df) > 1:
            df['Upload Time'] = pd.to_datetime(df['Upload Time'])
            timeline_data = df.groupby(df['Upload Time'].dt.date).size().reset_index()
            timeline_data.columns = ['Date', 'Documents Uploaded']
            
            fig = px.line(timeline_data, x='Date', y='Documents Uploaded', 
                         title='📅 Document Upload Timeline', 
                         markers=True, line_shape='spline')
            fig.update_traces(line=dict(color='#1f77b4', width=3))
            st.plotly_chart(fig, use_container_width=True)
        
        # Document list with details
        st.markdown("### 📋 Document Details")
        for i, meta in enumerate(self.kb.processor.document_metadata):
            with st.expander(f"{meta['filename']} - {meta['file_type']} - {meta['text_length']:,} chars"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Summary:**")
                    st.write(meta['summary'])
                with col2:
                    st.write("**Key Insights:**")
                    st.write(meta['insights'])
    
    def render_knowledge_base(self):
        """Render knowledge base explorer"""
        st.markdown("<h2 class='sub-header'>🧠 Knowledge Base Explorer</h2>", unsafe_allow_html=True)
        
        if not self.kb.processor.document_metadata:
            st.info("""
            ## 🧠 Knowledge Base is Empty!
            
            Your knowledge base will display all processed documents with their summaries and insights.
            
            **To get started:**
            1. Go to **'Upload Documents'**
            2. Upload your PDF, DOCX, or TXT files
            3. Wait for processing to complete
            4. Come back here to explore the insights!
            """)
            return
        
        # Document selector with auto-select from dashboard
        doc_names = [doc['filename'] for doc in self.kb.processor.document_metadata]
        default_index = 0
        if 'selected_doc' in st.session_state:
            if st.session_state.selected_doc in doc_names:
                default_index = doc_names.index(st.session_state.selected_doc)
        
        selected_doc = st.selectbox("Select a document to explore:", doc_names, index=default_index)
        
        if selected_doc:
            doc_index = doc_names.index(selected_doc)
            doc_meta = self.kb.processor.document_metadata[doc_index]
            doc_content = self.kb.processor.documents[doc_index]
            
            # Document header
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Filename", doc_meta['filename'])
            with col2:
                st.metric("File Type", doc_meta['file_type'])
            with col3:
                st.metric("Text Length", f"{doc_meta['text_length']:,} chars")
            with col4:
                st.metric("Uploaded", doc_meta['upload_time'].split()[0])
            
            st.markdown("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📋 Document Summary")
                st.markdown(f'<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">{doc_meta["summary"]}</div>', 
                           unsafe_allow_html=True)
                
                st.markdown("### 💡 Key Insights")
                st.markdown(f'<div style="background-color: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107;">{doc_meta["insights"]}</div>', 
                           unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📝 Document Preview")
                preview_text = doc_content[:2000] + "..." if len(doc_content) > 2000 else doc_content
                st.text_area(
                    "Content Preview", 
                    preview_text, 
                    height=400,
                    label_visibility="collapsed"
                )
                
                # Download processed insights
                insights_data = {
                    'filename': doc_meta['filename'],
                    'upload_time': doc_meta['upload_time'],
                    'summary': doc_meta['summary'],
                    'insights': doc_meta['insights'],
                    'text_length': doc_meta['text_length']
                }
                
                st.download_button(
                    label="📥 Download Insights as JSON",
                    data=json.dumps(insights_data, indent=2),
                    file_name=f"{doc_meta['filename']}_insights.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    def run(self):
        """Main application runner"""
        page = self.render_sidebar()
        
        if page == "🏠 Dashboard":
            self.render_dashboard()
        elif page == "📤 Upload Documents":
            self.render_upload()
        elif page == "🔍 Search & Query":
            self.render_search()
        elif page == "📊 Analytics":
            self.render_analytics()
        elif page == "🧠 Knowledge Base":
            self.render_knowledge_base()

# Run the application
if __name__ == "__main__":
    app = DocumentAgentApp()
    app.run()