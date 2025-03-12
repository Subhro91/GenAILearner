import os
import uuid
import shutil
import webbrowser
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import fitz  # PyMuPDF
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import atexit
import psutil
import traceback

app = FastAPI()

# Directory setup
UPLOAD_FOLDER = "uploaded_pdfs/"
MODELS_FOLDER = "models/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Store uploaded PDFs info
pdf_files = {}

# Flag to track if browser has been opened
browser_opened = False

def cleanup_python_processes():
    """Clean up any existing Python processes on exit"""
    current_process = psutil.Process()
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Kill other Python processes except the current one
            if proc.info['name'] == 'python.exe' and proc.pid != current_process.pid:
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

# Register cleanup function
atexit.register(cleanup_python_processes)

# Check if CUDA is available and set device
device = 0 if torch.cuda.is_available() else -1  # Use integer device IDs for pipeline
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Initialize AI models with GPU support if available
try:
    # Try to load models from local directory first
    if os.path.exists(os.path.join(MODELS_FOLDER, "summarizer")):
        print("Loading summarization model from local directory...")
        summarizer = pipeline("summarization", 
                             model=os.path.join(MODELS_FOLDER, "summarizer"),
                             max_length=65,
                             min_length=30,
                             device=device)
    else:
        print("Downloading summarization model from Hugging Face...")
        summarizer = pipeline("summarization", 
                             model="sshleifer/distilbart-cnn-12-6", 
                             max_length=65,
                             min_length=30,
                             device=device)
    
    # Load the model and tokenizer separately
    model_name = "google/flan-t5-large"
    if os.path.exists(os.path.join(MODELS_FOLDER, "flan-t5-tokenizer")) and os.path.exists(os.path.join(MODELS_FOLDER, "flan-t5-model")):
        print("Loading question generation model from local directory...")
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODELS_FOLDER, "flan-t5-tokenizer"))
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(MODELS_FOLDER, "flan-t5-model"))
    else:
        print("Downloading question generation model from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Create the pipeline with the configured model and tokenizer
    qa_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        device=device
    )
    
    print("Using models:")
    print(f"- Summarization: sshleifer/distilbart-cnn-12-6 (distilled model) on {'cuda' if device >= 0 else 'cpu'}")
    print(f"- Quiz Generation: google/flan-t5-large on {'cuda' if device >= 0 else 'cpu'}")
except Exception as e:
    print(f"ERROR loading models: {str(e)}")
    print("Please run download_models.py first to download the required AI models.")
    print("The application will continue to run, but some features may not work correctly.")

@app.on_event("startup")
async def startup_event():
    """Open browser when the application starts"""
    global browser_opened
    if not browser_opened:
        cleanup_python_processes()  # Clean up existing processes
        webbrowser.open("http://127.0.0.1:8000")
        browser_opened = True

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF upload"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Store file info
    pdf_id = str(uuid.uuid4())
    pdf_files[pdf_id] = {
        "filename": file.filename,
        "path": file_path
    }
    
    return {"pdf_id": pdf_id, "filename": file.filename}

@app.get("/process/{pdf_id}")
async def process_pdf(pdf_id: str):
    """Extract text and generate summary from PDF"""
    if pdf_id not in pdf_files:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    file_path = pdf_files[pdf_id]["path"]
    
    try:
        text = ""
        print(f"Opening PDF file: {file_path}")
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        print(f"Extracted text length: {len(text)}")
        
        # Split text into smaller chunks (500 tokens) for better summarization
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter out small chunks
        
        print(f"Processing {len(chunks)} chunks")
        
        # Process chunks in batches for better GPU utilization
        batch_size = 4  # Adjust based on your GPU memory
        summaries = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            try:
                batch_summaries = summarizer(
                    batch,
                    max_length=130,
                    min_length=30,
                    do_sample=False,
                    truncation=True,
                    batch_size=len(batch)
                )
                # Correctly extract summary text from each item in batch_summaries
                for summary in batch_summaries:
                    summaries.append(summary['summary_text'])
            except Exception as e:
                print(f"Error summarizing batch: {str(e)}")
                continue
        
        if not summaries:
            raise HTTPException(status_code=500, detail="Could not generate summary from the text")
        
        final_summary = " ".join(summaries)
        print(f"Final summary length: {len(final_summary)}")
        
        return {
            "pdf_id": pdf_id,
            "full_text": text,
            "summary": final_summary
        }
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/generate-quiz/{pdf_id}")
async def generate_quiz(pdf_id: str, num_questions: int = 10, difficulty: str = "medium"):
    """Generate quiz questions based on the PDF content with specified difficulty"""
    if pdf_id not in pdf_files:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    try:
        # First get the processed content
        content = await process_pdf(pdf_id)
        summary = content["summary"]
        
        if not summary:
            raise HTTPException(status_code=400, detail="No summary available to generate quiz")
        
        # Set the difficulty based on user selection - only generate questions of selected difficulty
        difficulty = difficulty.lower()
        selected_difficulty = difficulty.capitalize()
        
        # Extract key topics and facts from the summary to use in questions
        topics = []
        sentences = [s.strip() for s in summary.split('.') if len(s.strip()) > 20]
        if len(sentences) >= 3:
            topics = sentences[:min(len(sentences), num_questions * 2)]  # Get twice as many topics as needed
        
        # Determine if the content is narrative/story-like or informational/technical
        is_narrative = is_narrative_content(summary)
        
        # Create a focused prompt with a shorter summary to avoid token length issues
        prompt = f"""Generate {num_questions} multiple-choice questions directly based on the following {'story' if is_narrative else 'text'} summary.
All questions must be {selected_difficulty} difficulty level.

{'STORY' if is_narrative else 'TEXT'} CONTENT:
{summary[:800]}

INSTRUCTIONS:
1. Create simple, straightforward questions that directly reference specific details, facts, or concepts from the summary
2. ONLY create questions about information explicitly mentioned in the summary
3. Each question must have 4 options (A, B, C, D)
4. Only one option should be correct, and it must be information from the summary
5. The other three options should be plausible but incorrect
6. Vary which option is correct (A, B, C, or D)
7. Include the correct answer after each question
8. Keep questions clear and concise

Example format:
What is the main topic discussed in the text?
A) [Topic actually mentioned in the summary]
B) [Plausible but incorrect topic]
C) [Plausible but incorrect topic]
D) [Plausible but incorrect topic]
Answer: A
"""

        print(f"Generating quiz with difficulty: {difficulty}")
        
        # Use the pipeline with a smaller max_length to avoid token length issues
        generated_text = qa_pipeline(
            prompt,
            max_length=1024,
            do_sample=True,
            temperature=0.6,  # Slightly lower temperature for more focused responses
            top_p=0.9
        )[0]['generated_text']
        
        # Process the generated text to extract questions
        all_questions = []
        question_text = ""
        options = []
        correct_answer = None
        explanation = ""
        
        # Split the text by lines and process
        lines = generated_text.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # If this is a potential question (not starting with A), B), etc.)
            if not line.startswith(("A)", "B)", "C)", "D)", "Answer:")) and "?" in line:
                # If we have a previous question, save it
                if question_text and len(options) == 4 and correct_answer is not None:
                    all_questions.append({
                        'question': question_text,
                        'options': options,
                        'correct_answer': correct_answer,
                        'difficulty': selected_difficulty,
                        'explanation': explanation or f"This is based on the {is_narrative and 'story' or 'text'} content."
                    })
                
                # Start a new question
                question_text = line
                options = []
                correct_answer = None
                explanation = ""
            
            # Process options
            elif line.startswith(("A)", "B)", "C)", "D)")) and len(line) > 3:
                option_text = line[3:].strip()  # Changed from [2:] to [3:] to handle "A) " format
                options.append(option_text)
            
            # Process answer
            elif line.startswith("Answer:") and len(line) > 8:
                answer_letter = line[8:].strip()  # Changed from [7:] to [8:] to handle "Answer: " format
                if answer_letter in ["A", "B", "C", "D"]:
                    correct_answer = ord(answer_letter) - ord('A')
        
        # Add the last question
        if question_text and len(options) == 4 and correct_answer is not None:
            all_questions.append({
                'question': question_text,
                'options': options,
                'correct_answer': correct_answer,
                'difficulty': selected_difficulty,
                'explanation': explanation or f"This is based on the {is_narrative and 'story' or 'text'} content."
            })
        
        # If we don't have enough questions, use fallback questions
        if len(all_questions) < num_questions:
            # Create fallback questions based on the content
            for i in range(len(all_questions), num_questions):
                correct_pos = i % 4  # Distribute correct answers evenly
                
                # Use actual sentences from the summary for fallback questions
                if len(sentences) > i:
                    # Extract a sentence to use as the basis for a question
                    sentence = sentences[i]
                    
                    # Create a simple question from the sentence
                    if "?" in sentence:
                        # If the sentence is already a question, use it
                        question = sentence
                    else:
                        # Create a "what" question about the sentence
                        first_few_words = " ".join(sentence.split()[:3])
                        question = f"What does the text say about {first_few_words}...?"
                    
                    # Create options with the correct one being from the text
                    options = [""] * 4
                    
                    # The correct option contains information from the sentence
                    options[correct_pos] = sentence
                    
                    # Create plausible but incorrect options
                    for j in range(4):
                        if j != correct_pos:
                            if len(sentences) > i + j + 1:
                                # Use other sentences but modify them slightly
                                other_sentence = sentences[i + j + 1]
                                words = other_sentence.split()
                                if len(words) > 5:
                                    # Swap some words to make it incorrect but plausible
                                    if len(words) > 10:
                                        temp = words[3]
                                        words[3] = words[8]
                                        words[8] = temp
                                    options[j] = " ".join(words)
                                else:
                                    options[j] = f"The text does not mention {other_sentence}"
                            else:
                                options[j] = f"This information is not found in the text"
                else:
                    # If we don't have enough sentences, use generic fallback
                    if is_narrative:
                        question = get_narrative_fallback_question(i)
                        options = get_narrative_fallback_options(i, correct_pos)
                    else:
                        question = get_informational_fallback_question(i)
                        options = get_informational_fallback_options(i, correct_pos)
                
                all_questions.append({
                    'number': i + 1,
                    'question': question,
                    'options': options,
                    'correct_answer': correct_pos,
                    'difficulty': selected_difficulty,
                    'explanation': f"This is based on the {is_narrative and 'story' or 'text'} content."
                })
        
        # Number the questions
        for i, q in enumerate(all_questions):
            q['number'] = i + 1
        
        print(f"Generated {len(all_questions)} questions for difficulty: {difficulty}")
        
        # Return the questions
        return {"questions": all_questions[:num_questions]}
        
    except Exception as e:
        print(f"Error generating quiz: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

# Helper function to check if content is narrative/story-like
def is_narrative_content(text):
    """Determine if the text is narrative (story-like) or informational/technical"""
    # Convert to lowercase for comparison
    text_lower = text.lower()
    
    # Check for narrative indicators
    narrative_indicators = [
        # Character indicators
        "character", "protagonist", "hero", "heroine", "villain", 
        # Names with pronouns nearby
        r"\b[A-Z][a-z]+\b.{1,20}\b(he|she|they|his|her|their)\b",
        # Story elements
        "story", "tale", "adventure", "journey", "quest",
        # Dialogue indicators
        '"', "'", "said", "asked", "replied", "exclaimed",
        # Plot elements
        "plot", "chapter", "scene", "setting", "beginning", "ending",
        # Common narrative phrases
        "once upon a time", "one day", "long ago", "in a land", 
        "suddenly", "eventually", "finally", "in the end"
    ]
    
    # Check for informational indicators
    informational_indicators = [
        # Academic/technical indicators
        "study", "research", "analysis", "data", "method", "results",
        "conclusion", "findings", "evidence", "hypothesis", "theory",
        # Factual indicators
        "fact", "statistic", "percent", "figure", "table", "chart",
        # Organizational indicators
        "section", "chapter", "part", "introduction", "summary",
        # Reference indicators
        "according to", "reference", "citation", "source", "bibliography",
        # Technical terms
        "technical", "scientific", "academic", "professional"
    ]
    
    # Count indicators
    narrative_count = 0
    informational_count = 0
    
    import re
    
    # Check for narrative indicators
    for indicator in narrative_indicators:
        if indicator.startswith(r"\b"):  # It's a regex pattern
            matches = re.findall(indicator, text_lower)
            narrative_count += len(matches)
        else:
            narrative_count += text_lower.count(indicator)
    
    # Check for informational indicators
    for indicator in informational_indicators:
        informational_count += text_lower.count(indicator)
    
    # Additional checks for narrative structure
    # Check for character names (capitalized words not at the start of sentences)
    potential_names = re.findall(r'(?<!\.)\s+([A-Z][a-z]+)', text)
    if len(potential_names) > 3:  # Multiple potential character names
        narrative_count += len(potential_names)
    
    # Check for past tense verbs, common in narratives
    past_tense_verbs = re.findall(r'\b(was|were|had|went|came|saw|heard|felt|thought|said|did|made)\b', text_lower)
    narrative_count += len(past_tense_verbs) // 3  # Don't count each one fully
    
    # Determine the type based on the counts
    return narrative_count > informational_count

# Helper function to check if a question is natural-sounding
def is_natural_question(question):
    """Check if a question sounds natural and conversational"""
    # Convert to lowercase for comparison
    q_lower = question.lower()
    
    # Check for overly formal or generic phrasings to avoid
    generic_starts = [
        "according to the text, what",
        "what does the text say about",
        "what does the document state",
        "as mentioned in the text",
        "the text states that",
        "which statement from the text",
        "which of the following does the text"
    ]
    
    # Check if the question starts with any of these generic phrases
    for start in generic_starts:
        if q_lower.startswith(start):
            # Allow some exceptions if the question continues with specific content
            if len(q_lower) > len(start) + 30:  # If it's a longer, potentially more specific question
                continue
            return False
    
    # Check if the question is incomplete (cut off)
    if len(question) > 10 and not question.endswith("?"):
        return False
    
    # Check for questions that are too short
    if len(question.split()) < 4:
        return False
    
    # Default to accepting the question
    return True

# Helper functions for creating narrative-style questions and options
def create_narrative_question(topic, index):
    """Create a natural-sounding question for narrative content"""
    question_templates = [
        f"What happens after {topic[:20]}...?",
        f"How does the character react when {topic[:20]}...?",
        f"Why does the protagonist {topic[:20]}...?",
        f"What discovery is made in the scene where {topic[:20]}...?",
        f"What challenge does the main character face when {topic[:20]}...?",
        f"What is revealed about the setting when {topic[:20]}...?",
        f"How does the relationship change after {topic[:20]}...?",
        f"What motivates the character to {topic[:20]}...?",
        f"What surprise awaits the protagonist when {topic[:20]}...?",
        f"What decision must be made after {topic[:20]}...?"
    ]
    return question_templates[index % len(question_templates)]

def create_informational_question(topic, index):
    """Create a natural-sounding question for informational content"""
    question_templates = [
        f"What conclusion does the text draw about {topic[:20]}...?",
        f"How does {topic[:20]}... affect the overall argument?",
        f"What evidence supports the claim that {topic[:20]}...?",
        f"Why is {topic[:20]}... significant in this context?",
        f"What relationship exists between {topic[:20]}... and other factors?",
        f"How does the author explain the process of {topic[:20]}...?",
        f"What distinction is made regarding {topic[:20]}...?",
        f"What principle underlies the concept of {topic[:20]}...?",
        f"How does {topic[:20]}... contribute to our understanding?",
        f"What pattern emerges from the analysis of {topic[:20]}...?"
    ]
    return question_templates[index % len(question_templates)]

def create_narrative_correct_option(topic):
    """Create a natural-sounding correct option for narrative content"""
    return f"The story reveals that {topic}"

def create_informational_correct_option(topic):
    """Create a natural-sounding correct option for informational content"""
    return f"The text explains that {topic}"

def create_narrative_incorrect_option(topic, index):
    """Create plausible but incorrect options for narrative content"""
    incorrect_templates = [
        f"The character avoids this situation entirely",
        f"This event happens to a different character",
        f"This occurs much later in the story",
        f"The protagonist dreams this rather than experiencing it",
        f"This is merely hinted at but never actually happens",
        f"This is prevented by another character's intervention",
        f"The opposite actually occurs in the story",
        f"This is a misinterpretation of the character's actions"
    ]
    return incorrect_templates[index % len(incorrect_templates)]

def create_informational_incorrect_option(topic, index):
    """Create plausible but incorrect options for informational content"""
    incorrect_templates = [
        f"The text presents this as a common misconception",
        f"This contradicts the main evidence presented",
        f"The author specifically argues against this view",
        f"This represents an outdated understanding of the concept",
        f"This is mentioned but dismissed as improbable",
        f"The text identifies this as a limitation of current research",
        f"This is presented as a competing theory with less support",
        f"The text acknowledges this view but favors an alternative explanation"
    ]
    return incorrect_templates[index % len(incorrect_templates)]

# Helper functions for narrative fallback questions
def get_narrative_fallback_question(index):
    """Return varied fallback questions for narrative content"""
    questions = [
        "What motivates the main character's actions in this part of the story?",
        "How does the protagonist respond to the challenge presented?",
        "What significant event changes the course of the narrative?",
        "What revelation surprises the characters in this scene?",
        "How does the setting influence the events that unfold?",
        "What conflict emerges between the characters?",
        "What clue helps solve the mystery presented in the story?",
        "How does the main character's perspective shift?",
        "What obstacle prevents the protagonist from achieving their goal?",
        "What decision proves crucial to the story's outcome?"
    ]
    return questions[index % len(questions)]

def get_narrative_fallback_options(index, correct_pos):
    """Return fallback options for narrative content with the correct answer in the specified position"""
    # Base templates for options
    option_sets = [
        [
            "The character discovers an important truth that changes everything",
            "The character misinterprets the situation and makes a poor choice",
            "The character ignores advice and follows their own instincts",
            "The character relies on help from an unexpected ally"
        ],
        [
            "A hidden connection between characters is revealed",
            "A seemingly minor detail proves to be critically important",
            "A character's true intentions are exposed",
            "A surprising twist forces everyone to reconsider their plans"
        ],
        [
            "The protagonist finds strength in a moment of crisis",
            "The protagonist learns from a past mistake",
            "The protagonist misunderstands another character's motives",
            "The protagonist makes a sacrifice for someone else"
        ],
        [
            "An unexpected opportunity presents itself at a crucial moment",
            "A long-held secret threatens to unravel everything",
            "A chance encounter leads to a vital discovery",
            "A misunderstanding creates tension between allies"
        ],
        [
            "The character's unique skills prove essential to success",
            "The character's fear prevents them from seeing the truth",
            "The character's past experiences provide valuable insight",
            "The character's relationship with others is tested"
        ]
    ]
    
    # Get the base set for this index
    options = option_sets[index % len(option_sets)].copy()
    
    # Move the correct answer to the desired position
    correct_option = options[0]  # By convention, first option is "correct" in our templates
    options.pop(0)  # Remove it
    options.insert(correct_pos, correct_option)  # Insert at desired position
    
    return options[:4]

# Helper functions for informational fallback questions
def get_informational_fallback_question(index):
    """Return varied fallback questions for informational content"""
    questions = [
        "What key insight does the text provide about this topic?",
        "How does the author support the main argument?",
        "What distinction does the text make between related concepts?",
        "What evidence strengthens the central claim?",
        "How does the text explain this phenomenon?",
        "What factor is identified as most influential?",
        "What connection does the author establish between these elements?",
        "What principle underlies the approach described?",
        "How does the text challenge conventional understanding?",
        "What methodology yields the most significant findings?"
    ]
    return questions[index % len(questions)]

def get_informational_fallback_options(index, correct_pos):
    """Return fallback options for informational content with the correct answer in the specified position"""
    # Base templates for options
    option_sets = [
        [
            "By presenting multiple lines of evidence that converge on a single conclusion",
            "By relying primarily on theoretical models rather than empirical data",
            "By comparing competing interpretations without favoring any particular view",
            "By focusing on exceptions rather than general patterns"
        ],
        [
            "It reveals unexpected connections between seemingly unrelated factors",
            "It challenges the methodology used in previous studies",
            "It reframes the question in more productive terms",
            "It applies established principles to new contexts"
        ],
        [
            "The integration of multiple perspectives yields a more complete understanding",
            "The limitations of current methods prevent definitive conclusions",
            "The historical context explains current observations",
            "The practical applications outweigh theoretical concerns"
        ],
        [
            "Careful analysis of specific examples reveals broader patterns",
            "Statistical correlations suggest but don't confirm causation",
            "Qualitative assessments complement quantitative measurements",
            "Controlled experiments eliminate alternative explanations"
        ],
        [
            "By identifying the underlying mechanisms driving observed effects",
            "By documenting variations across different contexts and conditions",
            "By tracking changes over time to establish trends",
            "By isolating variables to determine their individual contributions"
        ]
    ]
    
    # Get the base set for this index
    options = option_sets[index % len(option_sets)].copy()
    
    # Move the correct answer to the desired position
    correct_option = options[0]  # By convention, first option is "correct" in our templates
    options.pop(0)  # Remove it
    options.insert(correct_pos, correct_option)  # Insert at desired position
    
    return options[:4]

@app.delete("/delete/{pdf_id}")
async def delete_pdf(pdf_id: str):
    """Delete a PDF file"""
    if pdf_id not in pdf_files:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    try:
        os.remove(pdf_files[pdf_id]["path"])
        del pdf_files[pdf_id]
        return {"message": "PDF deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



