from transformers import BartForConditionalGeneration, BartTokenizer, AutoTokenizer, AutoModel
#import os 
import fitz 
import gradio as gr

#Load models
sum_model_name = "facebook/bart-large-cnn"
sum_model = BartForConditionalGeneration.from_pretrained(sum_model_name)
sum_tokenizer = BartTokenizer.from_pretrained(sum_model_name)
doc_rank_model_name = "sentence-transformers/msmarco-distilbert-base-v3"
doc_rank_model = AutoModel.from_pretrained(doc_rank_model_name)
doc_rank_tokenizer = AutoTokenizer.from_pretrained(doc_rank_model_name)

#Warnungen von Transformers werden unterdrückt
#logging.set_verbosity_error() 

def extract_text_from_pdf(pdf_path): 
    with fitz.open(pdf_path) as doc: 
        return "".join([page.get_text() for page in doc])
    
def rank_documents(pdf_paths, question): 
    doc_scores = {}
    for pdf_path in pdf_paths: 
        text = extract_text_from_pdf(pdf_path)
        inputs = doc_rank_tokenizer(text, question, return_tensors="pt", truncation=True, padding=True)
        outputs = doc_rank_model(**inputs)
        doc_embeddings = outputs.last_hidden_state.mean(dim=1)
        doc_score = float(doc_embeddings.norm(dim=-1))
        doc_scores[pdf_path] = doc_score
    return doc_scores

def generate_summaries(top_docs, question): 
    summaries = {}
    for pdf_path in top_docs:
        doc_text = extract_text_from_pdf(pdf_path)
        inputs = sum_tokenizer.encode(question + " " + doc_text, return_tensors="pt", max_length=1024, truncation=True)
        sum_ids = sum_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summaries[pdf_path] = sum_tokenizer.decode(sum_ids[0], skip_special_tokens=True)
    return summaries

def process_pdfs(question, pdf_files): 
    pdf_paths = [pdf.name for pdf in pdf_files]
    doc_scores = rank_documents(pdf_paths, question)

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top_docs = [doc[0] for doc in sorted_docs]

    summaries = generate_summaries(top_docs, question) 

    return {"Ranked Documents": sorted_docs, "Summaries": summaries}

#Gradio Interface 
inputs = [
    gr.Textbox(lines=1, placeholder="Enter your question here", label="Question"), 
    gr.File(file_count="multiple", label="Upload your PDF Files")
]

outputs = gr.JSON(label="Results")

gr.Interface(fn=process_pdfs, inputs=inputs, outputs=outputs, title="Wardrobe Tool").launch()