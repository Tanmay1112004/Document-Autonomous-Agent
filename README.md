# Document Autonomous Agent 🤖

A powerful, autonomous document processing agent that can summarize, extract insights, and create a comprehensive knowledge base from your documents. Think "ChatGPT for your documents" but fully autonomous and continuously learning.

![Document Agent](https://img.shields.io/badge/Document-Agent-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Gemini AI](https://img.shields.io/badge/Gemini-AI-orange)

## Demo Images

![demo](https://github.com/Tanmay1112004/codespaces-blank/blob/main/screenshots/Screenshot%202025-11-20%20181416.png)
![demo](https://github.com/Tanmay1112004/codespaces-blank/blob/main/screenshots/Screenshot%202025-11-20%20181655.png)
![demo](https://github.com/Tanmay1112004/codespaces-blank/blob/main/screenshots/Screenshot%202025-11-20%20181710.png)
![demo](https://github.com/Tanmay1112004/codespaces-blank/blob/main/screenshots/Screenshot%202025-11-20%20182038.png)
![demo](https://github.com/Tanmay1112004/codespaces-blank/blob/main/screenshots/Screenshot%202025-11-20%20182115.png)

## 🌟 Features

### Core Capabilities
- **📤 Multi-format Support**: Process PDF, DOCX, and TXT files
- **🤖 Automatic Summarization**: Generate comprehensive summaries using Gemini AI
- **💡 Insight Extraction**: Identify key findings, concepts, and important information
- **🔍 Semantic Search**: Find relevant documents using AI-powered search
- **❓ Q&A System**: Ask questions about your documents and get intelligent answers
- **📊 Analytics Dashboard**: Visualize document statistics and insights
- **🧠 Knowledge Base**: Explore all processed documents with summaries and insights
- **🔄 Continuous Learning**: Automatically learns from new documents you upload

### Advanced Features
- **Vector Embeddings**: Uses Sentence Transformers for semantic understanding
- **FAISS Integration**: Efficient similarity search and document retrieval
- **Real-time Processing**: Live progress tracking during document processing
- **Session Persistence**: Maintains knowledge base across browser sessions
- **Export Capabilities**: Download insights and summaries as JSON

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd document-autonomous-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up your Gemini API key**
   - The app is pre-configured with a demo API key
   - For production use, replace the API key in `document_agent.py`:
   ```python
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```

4. **Run the application**
```bash
streamlit run document_agent.py
```

5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Start uploading and processing documents!

## 📁 Project Structure

```
document-autonomous-agent/
│
├── document_agent.py          # Main application file
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── (session data)            # Automatically created by Streamlit
```

## 🎯 Usage Guide

### 1. Upload Documents
- Navigate to the **📤 Upload Documents** section
- Drag and drop or select PDF, DOCX, or TXT files
- Click "Process Uploaded Documents"
- Watch real-time progress and processing status

### 2. Explore Knowledge Base
- Go to **🧠 Knowledge Base** to view all processed documents
- Select any document to see:
  - Comprehensive summary
  - Key insights and findings
  - Document preview
  - Downloadable insights

### 3. Ask Questions
- Use the **🔍 Search & Query** section
- Ask natural language questions about your documents
- Get AI-powered answers with source references
- Use semantic search to find similar documents

### 4. View Analytics
- Check the **📊 Analytics** dashboard for:
  - Document statistics
  - File type distribution
  - Upload timeline
  - Text length analysis

## 🔧 Technical Architecture

### Components

#### 1. Document Processor
- **Text Extraction**: Handles PDF, DOCX, and TXT formats
- **Error Handling**: Robust error management for various file types
- **Text Validation**: Ensures extracted text is meaningful

#### 2. Document Analyzer
- **Gemini AI Integration**: Uses Google's Gemini for NLP tasks
- **Summary Generation**: Creates comprehensive document summaries
- **Insight Extraction**: Identifies key information and patterns
- **Q&A Engine**: Answers questions based on document content

#### 3. Knowledge Base
- **Vector Storage**: FAISS for efficient similarity search
- **Semantic Embeddings**: Sentence Transformers for document encoding
- **Session Management**: Persistent storage across app sessions

#### 4. Streamlit Frontend
- **Modern UI**: Beautiful, responsive interface
- **Real-time Updates**: Live progress and status indicators
- **Interactive Components**: Charts, expanders, and dynamic content

### AI Models Used
- **Gemini 1.5 Flash**: For summarization, insights, and Q&A
- **Sentence Transformers**: For document embeddings and semantic search
- **FAISS**: For efficient vector similarity search

## 📊 Supported File Types

| Format | Extension | Features |
|--------|-----------|----------|
| PDF | `.pdf` | Text extraction, summarization, insights |
| Word Document | `.docx` | Text extraction, summarization, insights |
| Text File | `.txt` | Direct processing, summarization, insights |

## ⚙️ Configuration

### API Keys
The application requires a Gemini API key. Update it in `document_agent.py`:

```python
GEMINI_API_KEY = "your_gemini_api_key_here"
```

### Model Settings
- Default model: `gemini-1.5-flash`
- Alternative: `gemini-1.5-pro` (change in `DocumentAnalyzer` class)

### Processing Limits
- Maximum file size: 200MB per file
- Text extraction limit: ~12,000 characters for summarization
- Supported languages: Multi-language (depends on Gemini capabilities)

## 🛠️ Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Ensure you're using a valid Gemini API key
   - Check that the model name is correct (`gemini-1.5-flash`)

2. **PDF text extraction fails**
   - The PDF might be scanned/image-based
   - Try using text-based PDFs for best results
   - Consider OCR solutions for scanned documents

3. **Session state resets**
   - Streamlit may reset on code changes
   - Upload documents again if needed
   - Use the clear function to start fresh

4. **Slow processing**
   - Large documents take longer to process
   - Gemini API has rate limits
   - Consider breaking large documents into smaller parts

### Performance Tips
- Use text-based PDFs instead of scanned documents
- Break very large documents into smaller sections
- Process multiple small documents instead of one large document
- Ensure stable internet connection for Gemini API calls

## 🔮 Future Enhancements

### Planned Features
- [ ] **OCR Support**: Process scanned PDFs and images
- [ ] **Batch Processing**: Handle large document collections
- [ ] **Custom Models**: Fine-tuned models for specific domains
- [ ] **API Endpoints**: REST API for integration
- [ ] **User Management**: Multi-user support with authentication
- [ ] **Advanced Analytics**: Topic modeling, sentiment analysis
- [ ] **Export Options**: PDF reports, CSV exports
- [ ] **Collaboration Features**: Share knowledge bases with teams

### Technical Improvements
- [ ] **Database Integration**: Persistent storage beyond sessions
- [ ] **Caching**: Improved performance for repeated queries
- [ ] **Async Processing**: Background document processing
- [ ] **Docker Support**: Containerized deployment

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests, report bugs, or suggest new features.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add comments for complex logic
- Include error handling

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** for powerful natural language processing
- **Streamlit** for the excellent web application framework
- **FAISS** for efficient similarity search
- **Sentence Transformers** for semantic embeddings
- **PyPDF2** and **python-docx** for document processing

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Create an issue in the repository
4. Contact the development team

---

**Document Autonomous Agent** - Making your documents smarter, one upload at a time! 🚀
