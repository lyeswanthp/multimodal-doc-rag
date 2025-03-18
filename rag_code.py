import torch
from typing import List, Dict
import numpy as np
from pathlib import Path
from colpali_engine.models import ColPali, ColPaliProcessor
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
from transformers import AutoModelForCausalLM, AutoConfig
import base64
from io import BytesIO
from tqdm import tqdm

def get_device():
    """Get the appropriate device for PyTorch"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class EmbedData:
    _model_instance = None
    _processor_instance = None

    def __init__(self, embed_model_name="vidore/colpali-v1.2", batch_size=4):
        self.embed_model_name = embed_model_name
        self.device = get_device()
        print(f"Using device: {self.device}")
        self.batch_size = batch_size
        self.embeddings = []
        
        if EmbedData._model_instance is None:
            self._load_embed_model()
        else:
            print("Using cached model instance")
            self.embed_model = EmbedData._model_instance
            self.processor = EmbedData._processor_instance
        
    def _load_embed_model(self):
        print("Loading model for the first time...")
        EmbedData._model_instance = ColPali.from_pretrained(
            self.embed_model_name,
            torch_dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True,
            cache_dir="./Janus/hf_cache"
        ).to(torch.float32)
        
        EmbedData._processor_instance = ColPaliProcessor.from_pretrained(
            self.embed_model_name,
            cache_dir="./Janus/hf_cache"
        )
        
        self.embed_model = EmbedData._model_instance
        self.processor = EmbedData._processor_instance

    def _process_embedding(self, embedding):
        """Process embedding to ensure consistent shape and format"""
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().float().numpy()
        
        if len(embedding.shape) > 2:
            embedding = embedding[0]
        
        if len(embedding.shape) > 2:
            embedding = embedding.reshape(embedding.shape[0], -1)
        
        return embedding

    def _convert_tensor_types(self, inputs):
        """Convert tensor types appropriately for the model"""
        converted = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if 'indices' in k.lower() or 'ids' in k.lower() or 'mask' in k.lower():
                    converted[k] = v.long()
                else:
                    converted[k] = v.to(torch.float32)
            else:
                converted[k] = v
        return converted

    def get_query_embedding(self, query):
        with torch.no_grad():
            inputs = self.processor.process_queries([query])
            inputs = self._convert_tensor_types(inputs)
            inputs = {k: v.to(self.embed_model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            query_embedding = self.embed_model(**inputs)
            processed_embedding = self._process_embedding(query_embedding)
            print(f"Query embedding shape: {processed_embedding.shape}")
            return processed_embedding

    def generate_embedding(self, images):
        with torch.no_grad():
            inputs = self.processor.process_images(images)
            inputs = self._convert_tensor_types(inputs)
            inputs = {k: v.to(self.embed_model.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            image_embeddings = self.embed_model(**inputs)
            processed_embeddings = [self._process_embedding(emb) for emb in image_embeddings]
            print(f"Image embedding shape: {processed_embeddings[0].shape}")
            return processed_embeddings
            
    def embed(self, images):
        """Process a list of images and generate their embeddings"""
        self.images = images
        self.embeddings = []
        
        for batch_images in tqdm(batch_iterate(images, self.batch_size), desc="Generating embeddings"):
            batch_embeddings = self.generate_embedding(batch_images)
            self.embeddings.extend(batch_embeddings)
            
        return self.embeddings

class InMemoryVectorStore:
    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.payloads: List[Dict] = []
        
    def add_vectors(self, vectors: List[np.ndarray], payloads: List[Dict]):
        for vec, payload in zip(vectors, payloads):
            # Ensure vector is properly shaped for storage
            if len(vec.shape) > 2:
                vec = vec.reshape(-1, vec.shape[-1])
            elif len(vec.shape) == 1:
                vec = vec.reshape(1, -1)
            self.vectors.append(vec)
            self.payloads.append(payload)
            
    def search(self, query_vector: np.ndarray, limit: int = 4) -> List[Dict]:
        if not self.vectors:
            print("No vectors in store")
            return []
        
        # Ensure query vector is properly shaped
        if len(query_vector.shape) > 2:
            query_vector = query_vector.reshape(-1, query_vector.shape[-1])
        elif len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        print(f"Debug: Query vector shape after reshaping: {query_vector.shape}")
        
        similarities = []
        for i, vec in enumerate(self.vectors):
            try:
                print(f"Debug: Vector {i} shape: {vec.shape}")
                
                # Compute mean embedding if we have multiple vectors
                if vec.shape[0] > 1:
                    vec = np.mean(vec, axis=0, keepdims=True)
                if query_vector.shape[0] > 1:
                    query_vector = np.mean(query_vector, axis=0, keepdims=True)
                
                # Ensure final shapes are correct
                vec = vec.reshape(1, -1)
                query_vector = query_vector.reshape(1, -1)
                
                # Compute cosine similarity
                dot_product = np.dot(vec.flatten(), query_vector.flatten())
                vec_norm = np.linalg.norm(vec)
                query_norm = np.linalg.norm(query_vector)
                
                if vec_norm == 0 or query_norm == 0:
                    sim = 0.0
                else:
                    sim = dot_product / (vec_norm * query_norm)
                
                # Convert to scalar
                sim = float(np.asarray(sim).item())
                similarities.append(sim)
                
            except Exception as e:
                print(f"Warning: Error computing similarity for vector {i}: {str(e)}")
                print(f"Vector shape: {vec.shape}, Query shape: {query_vector.shape}")
                similarities.append(0.0)
        
        if not similarities:
            print("No valid similarities computed")
            return []
        
        # Get top k results
        similarities = np.array(similarities)
        top_k = min(limit, len(similarities))
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'id': int(idx),
                'score': float(similarities[idx]),
                'payload': self.payloads[idx]
            })
        
        return results

class Retriever:
    def __init__(self, embeddata):
        self.embeddata = embeddata
        self.vector_store = InMemoryVectorStore()
        self._initialize_vector_store()
        
    def _initialize_vector_store(self):
        if not hasattr(self.embeddata, 'images') or not self.embeddata.images:
            print("Warning: No images available in embeddata")
            return
            
        # Ensure embeddings are properly shaped
        embeddings = []
        for emb in self.embeddata.embeddings:
            if len(emb.shape) == 1:
                emb = emb.reshape(1, -1)
            embeddings.append(emb)
            
        payloads = [
            {"image": image_to_base64(img)} 
            for img in self.embeddata.images
        ]
        
        if len(payloads) != len(embeddings):
            print(f"Warning: Mismatch between payloads ({len(payloads)}) and embeddings ({len(embeddings)})")
            return
            
        self.vector_store.add_vectors(embeddings, payloads)
        print(f"Initialized vector store with {len(embeddings)} vectors")

    def search(self, query):
        query_embedding = self.embeddata.get_query_embedding(query)
        print(f"Debug: Raw query embedding shape: {query_embedding.shape}")
        results = self.vector_store.search(query_embedding)
        
        if not results:
            print(f"No results found for query: {query}")
        else:
            print(f"Found {len(results)} results")
            
        return results

class RAG:
    _model_instance = None
    _processor_instance = None
    _tokenizer_instance = None

    def __init__(self, retriever, llm_name="deepseek-ai/Janus-Pro-1B"):
        self.llm_name = llm_name
        self.device = get_device()
        print(f"RAG using device: {self.device}")
        
        if RAG._model_instance is None:
            self._setup_llm()
        else:
            print("Using cached RAG model instance")
            self.vl_gpt = RAG._model_instance
            self.vl_chat_processor = RAG._processor_instance
            self.tokenizer = RAG._tokenizer_instance
            
        self.retriever = retriever

    def _setup_llm(self):
        print("Loading RAG model for the first time...")
        config = AutoConfig.from_pretrained(
            self.llm_name,
            cache_dir="./Janus/hf_cache"
        )
        config.torch_dtype = torch.float32
        
        RAG._processor_instance = VLChatProcessor.from_pretrained(
            self.llm_name, 
            cache_dir="./Janus/hf_cache"
        )
        self.vl_chat_processor = RAG._processor_instance
        RAG._tokenizer_instance = self.vl_chat_processor.tokenizer
        self.tokenizer = RAG._tokenizer_instance

        RAG._model_instance = AutoModelForCausalLM.from_pretrained(
            self.llm_name,
            config=config,
            trust_remote_code=True,
            cache_dir="./Janus/hf_cache",
            device_map=self.device,
        ).to(torch.float32).eval()
        
        self.vl_gpt = RAG._model_instance

    def _convert_tensor_types(self, inputs):
        """Convert tensor types appropriately for the model"""
        converted = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if 'indices' in k.lower() or 'ids' in k.lower() or 'mask' in k.lower():
                    converted[k] = v.long()
                else:
                    converted[k] = v.to(torch.float32)
            else:
                converted[k] = v
        return converted

    def generate_context(self, query):
        try:
            results = self.retriever.search(query)
            if not results:
                print("No matching results found for query")
                return None
                
            # Get all relevant image paths
            image_paths = []
            cache_dir = Path.cwd() / "cache" / "images"
            
            for result in results[:3]:  # Limit to top 3 results
                try:
                    image_id = int(result['id'])  # Ensure we have a proper integer
                    image_path = cache_dir / f"page{image_id}.jpg"
                    if image_path.exists():
                        image_paths.append(str(image_path.absolute()))
                    else:
                        print(f"Warning: Image file not found at {image_path}")
                except Exception as e:
                    print(f"Warning: Error processing result {result}: {str(e)}")
                    continue
                    
            return image_paths[0] if image_paths else None
            
        except Exception as e:
            print(f"Error in generate_context: {str(e)}")
            return None

    # Fix 2: Update the RAG.query method to handle multiple images
    def query(self, query):
        image_context = self.generate_context(query=query)
        if image_context is None:
            return "I apologize, but I couldn't find relevant content in the document for your query. Please try rephrasing your question."

        qa_prompt_tmpl_str = f"""The user has asked the following question:
                        Query: {query}
                        
                        Some images are available to you for this question. 
                        You have to understand these images thoroughly and 
                        extract all relevant information that will help you 
                        answer the query."""
        
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder> \n {qa_prompt_tmpl_str}",
                "images": [image_context] if isinstance(image_context, str) else image_context,
            },
            {
                "role": "Assistant",
                "content": "",
                "images": []
            },
        ]

        try:
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            )
            
            # Convert to dict and handle tensor types
            prepare_inputs_dict = {k: v for k, v in vars(prepare_inputs).items()}
            prepare_inputs_dict = self._convert_tensor_types(prepare_inputs_dict)
            
            # Ensure token indices are long type
            for key in ['input_ids', 'decoder_input_ids', 'attention_mask']:
                if key in prepare_inputs_dict:
                    prepare_inputs_dict[key] = prepare_inputs_dict[key].long()
            
            # Move to correct device
            prepare_inputs_dict = {
                k: v.to(self.vl_gpt.device) if isinstance(v, torch.Tensor) else v 
                for k, v in prepare_inputs_dict.items()
            }

            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs_dict)

            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs_dict['attention_mask'],
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )
            
            response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            return response

        except Exception as e:
            print(f"Error in query processing: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"