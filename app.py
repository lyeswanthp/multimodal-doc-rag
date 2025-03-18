import os
import gc
import uuid
import tempfile
from pathlib import Path
import gradio as gr
from pdf2image import convert_from_path
from rag_code import EmbedData, Retriever, RAG

# Create cache directory for storing processed images
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

class GradioMultimodalRAG:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.file_cache = {}
        self.messages = []
        self.query_engine = None
        
    def process_pdf(self, pdf_bytes):
        """Process uploaded PDF file and initialize RAG system"""
        if pdf_bytes is None:
            return "Please upload a PDF file.", None
            
        try:
            # Create temporary directory for PDF processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded bytes to a temporary file
                temp_pdf_path = os.path.join(temp_dir, "temp.pdf")
                with open(temp_pdf_path, "wb") as f:
                    f.write(pdf_bytes)
                
                file_key = f"{self.session_id}-document"
                
                if file_key not in self.file_cache:
                    print("Processing new PDF document...")
                    # Convert PDF to images
                    images = convert_from_path(temp_pdf_path)
                    
                    # Save pages as images
                    images_dir = CACHE_DIR / "images"
                    images_dir.mkdir(exist_ok=True)
                    
                    # Clean up any existing images
                    for existing_image in images_dir.glob('page*.jpg'):
                        existing_image.unlink()
                    
                    # Save new images
                    for i, image in enumerate(images):
                        image_path = images_dir / f'page{i}.jpg'
                        image.save(image_path, 'JPEG')
                        print(f"Saved image: {image_path}")
                    
                    # Initialize RAG system
                    print("Initializing RAG system...")
                    embeddata = EmbedData()
                    embeddata.embed(images)
                    
                    retriever = Retriever(embeddata=embeddata)
                    self.query_engine = RAG(retriever=retriever)
                    self.file_cache[file_key] = self.query_engine
                    
                    print("PDF processing completed successfully")
                else:
                    print("Using cached document processing")
                    self.query_engine = self.file_cache[file_key]
                
                return "PDF processed successfully! Ready to chat.", "Document loaded"
                
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return f"Error processing PDF: {str(e)}", None
    
    def chat(self, message, history):
        """Handle chat interactions with the RAG system"""
        if self.query_engine is None:
            history = history or []
            history.append((message, "Please upload and process a PDF document first."))
            return history
            
        try:
            print(f"Processing query: {message}")
            response = self.query_engine.query(message)
            history = history or []
            history.append((message, response))
            return history
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            history = history or []
            history.append((message, f"Error generating response: {str(e)}"))
            return history
    
    def clear_chat(self):
        """Reset the chat history"""
        self.messages = []
        return []
    
    def create_interface(self):
        """Create and configure the Gradio interface"""
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown(
                """
                # 🤖 Multimodal RAG with DeepSeek Janus
                Upload a PDF document and chat with an AI about its contents!
                
                ## Instructions:
                1. Upload a PDF document using the upload button
                2. Wait for the document to be processed
                3. Ask questions about the document's content
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        type="binary"
                    )
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                    filename_text = gr.Textbox(
                        label="Current File",
                        interactive=False,
                        visible=True
                    )
                    
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=600,
                        show_copy_button=True
                    )
                    msg = gr.Textbox(
                        label="Type your message",
                        placeholder="Ask me anything about the document...",
                        lines=2
                    )
                    with gr.Row():
                        submit = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear Chat")
            
            # Set up event handlers
            file_upload.upload(
                fn=self.process_pdf,
                inputs=[file_upload],
                outputs=[status_text, filename_text]
            )
            
            submit.click(
                fn=self.chat,
                inputs=[msg, chatbot],
                outputs=[chatbot],
                api_name="submit"
            ).then(
                fn=lambda: "",
                outputs=[msg]
            )
            
            clear.click(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot]
            )
            
            # Allow pressing Enter to submit
            msg.submit(
                fn=self.chat,
                inputs=[msg, chatbot],
                outputs=[chatbot]
            ).then(
                fn=lambda: "",
                outputs=[msg]
            )
            
        return interface

def main():
    # Configure environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
    
    # Create and launch the application
    app = GradioMultimodalRAG()
    interface = app.create_interface()
    
    # Launch with queuing enabled
    interface.queue()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

if __name__ == "__main__":
    main()