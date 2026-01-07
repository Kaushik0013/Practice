//3 Sum of numbers 1 to N (input from user)

import java.util.Scanner;
class Sum{
    public static void main(String[]args){
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter the number: ");
        //receiving input from th user
        int num = sc.nextInt();
        //taking the initial value of sum as zero
        int sum=0;
        //using for loop and finding out the sum 
        for(int i=0;i<=num;i++){
            sum+=i;
        }
        System.out.println(sum);
        sc.close();
    }
}



import streamlit as st
from datetime import date
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import uuid
import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# -------------------- ENV + DB SETUP --------------------
load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )

# -------------------- DB FUNCTIONS --------------------
def save_applicant_data(
    first_name, middle_name, last_name, dob, gender, marital_status,
    phone, email, aadhaar, pan, current_address, permanent_address
):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO loan_applicants (
                first_name, middle_name, last_name, dob, gender, marital_status,
                phone, email, aadhaar, pan, current_address, permanent_address
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            first_name, middle_name, last_name, dob, gender, marital_status,
            phone, email, aadhaar, pan, current_address, permanent_address
        ))

        conn.commit()
        cursor.close()
        conn.close()
        return True, "Details saved successfully!"

    except Exception as e:
        return False, str(e)

def save_chat_message(session_id, role, message):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO chat_history (session_id, role, message)
        VALUES (%s, %s, %s)
    """, (session_id, role, message))

    conn.commit()
    cursor.close()
    conn.close()

def load_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT role, message
        FROM chat_history
        WHERE session_id = %s
        ORDER BY created_at
    """, (session_id,))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return [{"role": r[0], "content": r[1]} for r in rows]

# -------------------- STREAMLIT SETUP --------------------
st.set_page_config(page_title="Loan Application with RAG", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history(st.session_state.session_id)

# -------------------- MODEL + RAG --------------------
@st.cache_resource
def load_model():
    model_path = "gemma"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    return model, tokenizer

@st.cache_resource
def load_faq_system():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection("loan_faqs")

    if collection.count() == 0:
        with open("card_activation_faqs.json", "r", encoding="utf-8") as f:
            faq_data = json.load(f)

        for faq in faq_data["faqs"]:
            text = f"Q: {faq['question']}\nA: {faq['answer']}"
            embedding = embedding_model.encode(text).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[text],
                ids=[faq["id"]]
            )

    return embedding_model, collection

# -------------------- CHAT RESPONSE --------------------
def generate_response(user_input, form_context):
    model, tokenizer = load_model()
    embedding_model, faq_collection = load_faq_system()

    query_embedding = embedding_model.encode(user_input).tolist()
    results = faq_collection.query(query_embeddings=[query_embedding], n_results=3)
    faq_context = "\n".join(results["documents"][0]) if results["documents"] else ""

    prompt = f"""
Applicant Info:
{form_context}

FAQ Info:
{faq_context}

User Question:
{user_input}
"""

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(input_ids["input_ids"], max_new_tokens=300)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# -------------------- FORM CONTEXT --------------------
def get_form_context(first_name, last_name, dob, gender, marital_status,
                     phone, email, aadhaar, pan, current_address, permanent_address):

    context = []
    if first_name: context.append(f"First name: {first_name}")
    if last_name: context.append(f"Last name: {last_name}")
    if dob: context.append(f"Age: {(date.today() - dob).days // 365}")
    if aadhaar: context.append("Aadhaar provided")
    if pan: context.append("PAN provided")

    return "\n".join(context) if context else "No form data yet."

# -------------------- UI --------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.title("Personal Details")

    first_name = st.text_input("First Name")
    middle_name = st.text_input("Middle Name")
    last_name = st.text_input("Last Name")
    dob = st.date_input("Date of Birth", value=None)
    gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])
    marital_status = st.selectbox("Marital Status", ["Select", "Single", "Married"])
    phone = st.text_input("Phone")
    email = st.text_input("Email")
    aadhaar = st.text_input("Aadhaar")
    pan = st.text_input("PAN")
    current_address = st.text_area("Current Address")
    permanent_address = st.text_area("Permanent Address")

    if st.button("Save Details"):
        success, msg = save_applicant_data(
            first_name, middle_name, last_name, dob, gender, marital_status,
            phone, email, aadhaar, pan, current_address, permanent_address
        )
        st.success(msg) if success else st.error(msg)

with right_col:
    st.subheader("Loan Assistant")

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask something...")
    if user_input:
        form_context = get_form_context(
            first_name, last_name, dob, gender, marital_status,
            phone, email, aadhaar, pan, current_address, permanent_address
        )

        response = generate_response(user_input, form_context)

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        save_chat_message(st.session_state.session_id, "user", user_input)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        save_chat_message(st.session_state.session_id, "assistant", response)

        st.rerun()


##
import streamlit as st
from datetime import date
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import uuid
import os
import psycopg2
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
 
load_dotenv()
 
st.set_page_config(page_title="Loan Application with RAG", layout="wide")
 

 
if "session_id" not in st.session_state:
    st.session_state.session_id= str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]
 
 
##PGSQL connection:
 
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )
 
##changes made
def save_applicant_data(
    first_name, middle_name, last_name, dob, gender, marital_status,
    phone, email, aadhaar, pan, current_address, permanent_address
):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO loan_applicants (
                first_name, middle_name, last_name, dob, gender, marital_status,
                phone, email, aadhaar, pan, current_address, permanent_address
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        values = (
            first_name, middle_name, last_name, dob, gender, marital_status,
            phone, email, aadhaar, pan, current_address, permanent_address
        )
        cursor.execute(query, values)
        applicant_id= cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        return True,applicant_id
    except Exception as e:
        return False, None
   
def save_chat_message(applicant_id,session_id, role, message):
    try:
        conn= get_db_connection()
        cursor= conn.cursor()
 
        cursor.execute(
            """
            INSERT INTO chat_history(
            applicant_id,
            session_id,
            role,
            message
            )
            VALUES(%s,%s,%s)
            """,
            (applicant_id,session_id, role, message)
        )
        conn.commit()
        cursor.close()
        conn.close()
 
    except Exception as e:
        st.error(f"Failed to save chat message: {e}")
 
def get_chat_history_from_db(session_id):
    conn= get_db_connection()
    cursor= conn.cursor()
 
    cursor.execute(
        """
        SELECT role, message, created_at
        FROM chat_history
        WHERE session_id = %s
        ORDER BY created_at
        """,
        (session_id,)
    )
    rows= cursor.fetchall()
    cursor.close()
    conn.close()
 
    return rows

def get_chat_history_by_applicant(applicant_id):
    conn= get_db_connection()
    cursor= conn.cursor()
 
    cursor.execute(
        """
        SELECT role, message, created_at
        FROM chat_history
        WHERE applicant_id = %s
        ORDER BY created_at
        """,
        (applicant_id,)
    )
    rows= cursor.fetchall()
    cursor.close()
    conn.close()
 
    return rows

# if 'chat_history' not in st.session_state:
#     db_rows= get_chat_history_from_db(st.session_state.session_id)

#     st.session_state.chat_history=[
#         {"role":role, "content":message}
#         for role, message,_ in db_rows
#     ]
 
    ## Loading FAQ System:
 
@st.cache_resource
def load_model():
    model_path = r"gemma"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    return model, tokenizer
 
@st.cache_resource
def load_faq_system():
    embedding_model = SentenceTransformer(r"all-MiniLM-L6-v2")
   
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection("loan_faqs")
   
    try:
        with open('card_activation_faqs.json', 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
    except FileNotFoundError:
        st.error("card_activation_faqs.json not found!")
        return embedding_model, collection
   
    if collection.count() == 0:
        for faq in faq_data.get('faqs', []):
            text = f"Q: {faq['question']}\nA: {faq['answer']}"
            embedding = embedding_model.encode(text).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "question": faq['question'],
                    "category": faq.get('category', ''),
                    "id": faq['id']
                }],
                ids=[faq['id']]
            )
        st.success(f"Loaded {len(faq_data.get('faqs', []))} FAQs into vector database")
   
    return embedding_model, collection
 
 
##Helper functions:
 
def get_form_context(first_name, last_name, dob, gender, marital_status,
                     phone, email, aadhaar, pan, current_address, permanent_address):
    context_parts = []
   
    if first_name:
        context_parts.append(f"Applicant's first name: {first_name}")
    if last_name:
        context_parts.append(f"Applicant's last name: {last_name}")
   
    if dob:
        age = (date.today() - dob).days // 365
        context_parts.append(f"Applicant's date of birth: {dob} (Age: {age} years)")
   
    if gender and gender != "Select Gender":
        context_parts.append(f"Gender: {gender}")
   
    if marital_status and marital_status != "Select Marital Status":
        context_parts.append(f"Marital status: {marital_status}")
   
    if phone:
        context_parts.append(f"Phone number provided: Yes")
   
    if email:
        context_parts.append(f"Email provided: Yes")
   
    if aadhaar:
        context_parts.append(f"Aadhaar card number provided: Yes")
   
    if pan:
        context_parts.append(f"PAN card number provided: Yes")
   
    if current_address:
        context_parts.append(f"Current address provided: Yes")
   
    if permanent_address:
        context_parts.append(f"Permanent address provided: Yes")
   
    missing = []
    if not first_name:
        missing.append("First name")
    if not last_name:
        missing.append("Last name")
    if not dob:
        missing.append("Date of birth")
    if not aadhaar:
        missing.append("Aadhaar card")
    if not pan:
        missing.append("PAN card")
   
    if missing:
        context_parts.append(f"Missing required fields: {', '.join(missing)}")
   
    return "\n".join(context_parts) if context_parts else "No form data filled yet."
 
def retrieve_relevant_faqs(user_input, embedding_model, faq_collection, top_k=3):
    try:
        query_embedding = embedding_model.encode(user_input).tolist()
       
        results = faq_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
       
        if results['documents'] and results['documents'][0]:
            return "\n\n".join(results['documents'][0])
        return ""
    except Exception as e:
        st.error(f"Error retrieving FAQs: {str(e)}")
        return ""
 
def generate_response(user_input, form_context):
    try:
        model, tokenizer = load_model()
        embedding_model, faq_collection = load_faq_system()
       
        faq_context = retrieve_relevant_faqs(user_input, embedding_model, faq_collection, top_k=3)
       
        system_prompt = f"""You are a helpful loan application assistant. Use the following information to provide accurate, personalized responses.
 
APPLICANT'S CURRENT FORM STATUS:
{form_context}
 
RELEVANT FAQ INFORMATION:
{faq_context}
 
Instructions:
- Answer based on the FAQ information when available
- Consider the applicant's form progress for personalized guidance
- If they ask about eligibility, use their age and provided details
- If asked what's needed, mention what they've provided and what's missing
- Be concise, helpful, and accurate"""
 
        messages = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "I'll provide personalized assistance based on the applicant's information and FAQ knowledge."}
        ]
       
        for msg in st.session_state.chat_history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
       
        messages.append({"role": "user", "content": user_input})
       
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        input_ids = input_ids.to(model.device)
       
        outputs = model.generate(
            input_ids,
            max_new_tokens=1000,
            temperature=0.3,
            do_sample=True,
            top_p=0.8,
            repetition_penalty=1.1
        )
       
        generated_ids = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        response = response.replace("<end_of_turn>", "").strip()
       
        return response if response else "I'm here to help! What would you like to know?"
    except Exception as e:
        return f"Error generating response: {str(e)}"
   
 
## UI:
 
left_col, right_col = st.columns([2, 1], gap="large")
 
with left_col:
    st.title("Personal Details")
    st.markdown("---")
   
    col1, col2, col3 = st.columns(3)
    with col1:
        first_name = st.text_input("First Name", placeholder="John")
    with col2:
        middle_name = st.text_input("Middle Name (Optional)", placeholder="Smith")
    with col3:
        last_name = st.text_input("Last Name", placeholder="Doe")
   
    col1, col2, col3 = st.columns(3)
    with col1:
        min_date = date.today().replace(year=date.today().year - 100)
        dob = st.date_input("Date of Birth", value=None, min_value=min_date, max_value=date.today())
    with col2:
        gender = st.selectbox("Gender", ["Select Gender", "Male", "Female", "Other", "Prefer not to say"])
    with col3:
        marital_status = st.selectbox("Marital Status", ["Select Marital Status", "Single", "Married", "Divorced", "Widowed"])
   
    col1, col2 = st.columns(2)
    with col1:
        phone = st.text_input("Phone Number", placeholder="(123) 456-7890")
    with col2:
        email = st.text_input("Email Address", placeholder="john.doe@example.com")
   
    col1, col2 = st.columns(2)
    with col1:
        aadhaar = st.text_input("Aadhaar Card Number", placeholder="XXXX XXXX XXXX")
    with col2:
        pan = st.text_input("PAN Card Number", placeholder="ABCDE1234F")
   
    st.markdown("**Current Address**")
    current_address = st.text_area("Current Address", placeholder="Enter your current address", height=100, label_visibility="collapsed")
   
    same_address = st.checkbox("Permanent Address is the same as Current Address")
   
    st.markdown("**Permanent Address**")
    if same_address:
        permanent_address = current_address
        st.text_area("Permanent Address", value=current_address, height=100, disabled=True, label_visibility="collapsed")
    else:
        permanent_address = st.text_area("Permanent Address", placeholder="Enter your permanent address", height=100, label_visibility="collapsed")
   
    st.markdown("")
 
    ##changes made
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Save Details", use_container_width=True):
            success, applicant_id = save_applicant_data(
                first_name,middle_name,last_name,dob,gender,marital_status,
                phone, email, aadhaar, pan, current_address, permanent_address
            )
 
            if success:
                st.session_state.applicant_id=applicant_id

                db_rows=get_chat_history_by_applicant(applicant_id)
                st.session_state.chat_history=[
                    {"role":role, "content":message}
                    for role, message,_ in db_rows
                ]
                st.success(f"Applicant ID: {applicant_id}")
            else:
                st.error("Rejected!")
 
with right_col:
    if st.button("View Chat History"):
        if "applicant_id" not in st.session_state:
            st.warning("Please save applicant details first.")
        else:
            chat_rows= get_chat_history_by_applicant(st.session_state.applicant_id)

            st.markdown(f"### Chat history for Applicant ID: {st.session_state.applicant_id}")
 
        if not chat_rows:
            st.info("No chat history found.")
        else:
            # st.markdown("Previous chat History:")
 
            for role,message, timestamp in chat_rows:
                with st.chat_message(role):
                    st.caption(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
                    st.write(message)
 
               
    st.markdown("### Your Loan Companion")
    chat_container = st.container(height=400, border=True)
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
   
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message...", label_visibility="collapsed")
        send_button = st.form_submit_button("Send", use_container_width=True, type="primary")
   
    if send_button and user_input:
        with st.spinner(" Retrieving FAQs and generating response..."):
            form_context = get_form_context(
                first_name, last_name, dob, gender, marital_status,
                phone, email, aadhaar, pan, current_address, permanent_address
            )
           
            bot_response = generate_response(user_input, form_context)
       
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        #change made
        save_chat_message(st.session_state.applicant_id,st.session_state.session_id, "user", user_input)
 
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
       
        save_chat_message(st.session_state.applicant_id,st.session_state.session_id, "assistant", bot_response)
 
 
       
        st.rerun()
 




















            ####
import streamlit as st
from datetime import date
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import uuid
import os
import psycopg2
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
 
load_dotenv()
 
st.set_page_config(page_title="Loan Application with RAG", layout="wide")
 

 
if "session_id" not in st.session_state:
    st.session_state.session_id= str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]
 
 
##PGSQL connection:
 
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )
 
##changes made
def save_applicant_data(
    first_name, middle_name, last_name, dob, gender, marital_status,
    phone, email, aadhaar, pan, current_address, permanent_address
):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO loan_applicants (
                first_name, middle_name, last_name, dob, gender, marital_status,
                phone, email, aadhaar, pan, current_address, permanent_address
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        values = (
            first_name, middle_name, last_name, dob, gender, marital_status,
            phone, email, aadhaar, pan, current_address, permanent_address
        )
        cursor.execute(query, values)
        applicant_id= cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        return True,applicant_id
    except Exception as e:
        return False, None
   
def save_chat_message(applicant_id,session_id, role, message):
    try:
        conn= get_db_connection()
        cursor= conn.cursor()
 
        cursor.execute(
            """
            INSERT INTO chat_history(
            applicant_id,
            session_id,
            role,
            message
            )
            VALUES(%s,%s,%s,%s)
            """,
            (applicant_id,session_id, role, message)
        )
        conn.commit()
        cursor.close()
        conn.close()
 
    except Exception as e:
        st.error(f"Failed to save chat message: {e}")
 
def get_chat_history_from_db(session_id):
    conn= get_db_connection()
    cursor= conn.cursor()
 
    cursor.execute(
        """
        SELECT role, message, created_at
        FROM chat_history
        WHERE session_id = %s
        ORDER BY created_at
        """,
        (session_id,)
    )
    rows= cursor.fetchall()
    cursor.close()
    conn.close()
 
    return rows

def get_chat_history_by_applicant(applicant_id):
    conn= get_db_connection()
    cursor= conn.cursor()
 
    cursor.execute(
        """
        SELECT role, message, created_at
        FROM chat_history
        WHERE applicant_id = %s
        ORDER BY created_at
        """,
        (applicant_id,)
    )
    rows= cursor.fetchall()
    cursor.close()
    conn.close()
 
    return rows

# if 'chat_history' not in st.session_state:
#     db_rows= get_chat_history_from_db(st.session_state.session_id)

#     st.session_state.chat_history=[
#         {"role":role, "content":message}
#         for role, message,_ in db_rows
#     ]
 
    ## Loading FAQ System:
 
@st.cache_resource
def load_model():
    model_path = r"gemma"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    return model, tokenizer
 
@st.cache_resource
def load_faq_system():
    embedding_model = SentenceTransformer(r"all-MiniLM-L6-v2")
   
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection("loan_faqs")
   
    try:
        with open('card_activation_faqs.json', 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
    except FileNotFoundError:
        st.error("card_activation_faqs.json not found!")
        return embedding_model, collection
   
    if collection.count() == 0:
        for faq in faq_data.get('faqs', []):
            text = f"Q: {faq['question']}\nA: {faq['answer']}"
            embedding = embedding_model.encode(text).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "question": faq['question'],
                    "category": faq.get('category', ''),
                    "id": faq['id']
                }],
                ids=[faq['id']]
            )
        st.success(f"Loaded {len(faq_data.get('faqs', []))} FAQs into vector database")
   
    return embedding_model, collection
 
 
##Helper functions:
 
def get_form_context(first_name, last_name, dob, gender, marital_status,
                     phone, email, aadhaar, pan, current_address, permanent_address):
    context_parts = []
   
    if first_name:
        context_parts.append(f"Applicant's first name: {first_name}")
    if last_name:
        context_parts.append(f"Applicant's last name: {last_name}")
   
    if dob:
        age = (date.today() - dob).days // 365
        context_parts.append(f"Applicant's date of birth: {dob} (Age: {age} years)")
   
    if gender and gender != "Select Gender":
        context_parts.append(f"Gender: {gender}")
   
    if marital_status and marital_status != "Select Marital Status":
        context_parts.append(f"Marital status: {marital_status}")
   
    if phone:
        context_parts.append(f"Phone number provided: Yes")
   
    if email:
        context_parts.append(f"Email provided: Yes")
   
    if aadhaar:
        context_parts.append(f"Aadhaar card number provided: Yes")
   
    if pan:
        context_parts.append(f"PAN card number provided: Yes")
   
    if current_address:
        context_parts.append(f"Current address provided: Yes")
   
    if permanent_address:
        context_parts.append(f"Permanent address provided: Yes")
   
    missing = []
    if not first_name:
        missing.append("First name")
    if not last_name:
        missing.append("Last name")
    if not dob:
        missing.append("Date of birth")
    if not aadhaar:
        missing.append("Aadhaar card")
    if not pan:
        missing.append("PAN card")
   
    if missing:
        context_parts.append(f"Missing required fields: {', '.join(missing)}")
   
    return "\n".join(context_parts) if context_parts else "No form data filled yet."
 
def retrieve_relevant_faqs(user_input, embedding_model, faq_collection, top_k=3):
    try:
        query_embedding = embedding_model.encode(user_input).tolist()
       
        results = faq_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
       
        if results['documents'] and results['documents'][0]:
            return "\n\n".join(results['documents'][0])
        return ""
    except Exception as e:
        st.error(f"Error retrieving FAQs: {str(e)}")
        return ""
 
def generate_response(user_input, form_context):
    try:
        model, tokenizer = load_model()
        embedding_model, faq_collection = load_faq_system()
       
        faq_context = retrieve_relevant_faqs(user_input, embedding_model, faq_collection, top_k=3)
       
        system_prompt = f"""You are a helpful loan application assistant. Use the following information to provide accurate, personalized responses.
 
APPLICANT'S CURRENT FORM STATUS:
{form_context}
 
RELEVANT FAQ INFORMATION:
{faq_context}
 
Instructions:
- Answer based on the FAQ information when available
- Consider the applicant's form progress for personalized guidance
- If they ask about eligibility, use their age and provided details
- If asked what's needed, mention what they've provided and what's missing
- Be concise, helpful, and accurate"""
 
        messages = [
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "I'll provide personalized assistance based on the applicant's information and FAQ knowledge."}
        ]
       
        for msg in st.session_state.chat_history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
       
        messages.append({"role": "user", "content": user_input})
       
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        input_ids = input_ids.to(model.device)
       
        outputs = model.generate(
            input_ids,
            max_new_tokens=1000,
            temperature=0.3,
            do_sample=True,
            top_p=0.8,
            repetition_penalty=1.1
        )
       
        generated_ids = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        response = response.replace("<end_of_turn>", "").strip()
       
        return response if response else "I'm here to help! What would you like to know?"
    except Exception as e:
        return f"Error generating response: {str(e)}"
   
 
## UI:
 
left_col, right_col = st.columns([2, 1], gap="large")
 
with left_col:
    st.title("Personal Details")
    st.markdown("---")
   
    col1, col2, col3 = st.columns(3)
    with col1:
        first_name = st.text_input("First Name", placeholder="John")
    with col2:
        middle_name = st.text_input("Middle Name (Optional)", placeholder="Smith")
    with col3:
        last_name = st.text_input("Last Name", placeholder="Doe")
   
    col1, col2, col3 = st.columns(3)
    with col1:
        min_date = date.today().replace(year=date.today().year - 100)
        dob = st.date_input("Date of Birth", value=None, min_value=min_date, max_value=date.today())
    with col2:
        gender = st.selectbox("Gender", ["Select Gender", "Male", "Female", "Other", "Prefer not to say"])
    with col3:
        marital_status = st.selectbox("Marital Status", ["Select Marital Status", "Single", "Married", "Divorced", "Widowed"])
   
    col1, col2 = st.columns(2)
    with col1:
        phone = st.text_input("Phone Number", placeholder="(123) 456-7890")
    with col2:
        email = st.text_input("Email Address", placeholder="john.doe@example.com")
   
    col1, col2 = st.columns(2)
    with col1:
        aadhaar = st.text_input("Aadhaar Card Number", placeholder="XXXX XXXX XXXX")
    with col2:
        pan = st.text_input("PAN Card Number", placeholder="ABCDE1234F")
   
    st.markdown("**Current Address**")
    current_address = st.text_area("Current Address", placeholder="Enter your current address", height=100, label_visibility="collapsed")
   
    same_address = st.checkbox("Permanent Address is the same as Current Address")
   
    st.markdown("**Permanent Address**")
    if same_address:
        permanent_address = current_address
        st.text_area("Permanent Address", value=current_address, height=100, disabled=True, label_visibility="collapsed")
    else:
        permanent_address = st.text_area("Permanent Address", placeholder="Enter your permanent address", height=100, label_visibility="collapsed")
   
    st.markdown("")
 
    ##changes made
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Save Details", use_container_width=True):
            success, applicant_id = save_applicant_data(
                first_name,middle_name,last_name,dob,gender,marital_status,
                phone, email, aadhaar, pan, current_address, permanent_address
            )
 
            if success:
                st.session_state.applicant_id=applicant_id

                db_rows=get_chat_history_by_applicant(applicant_id)
                st.session_state.chat_history=[
                    {"role":role, "content":message}
                    for role, message,_ in db_rows
                ]
                st.success(f"Applicant ID: {applicant_id}")
            else:
                st.error("Rejected!")
 
with right_col:
    if st.button("View Chat History"):
        if "applicant_id" not in st.session_state:
            conn= get_db_connection()
            cursor= conn.cursor()
            cursor.execute("SELECT id FROM loan_applicants ORDER BY id DESC")
            applicant_ids=[row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()

            if applicant_ids:
                 selected_id= st.selectbox(
                "Select Applicant ID",
                applicant_ids,
                key="applicant_selector"
                )
                 
                 if(
                     "applicant_id" not in st.session_state or st.session_state.applicant_id!=selected_id
                 ):
                     st.session_state.applicant_id= selected_id

                     db_rows= get_chat_history_by_applicant(selected_id)
                     st.session_state.chat_history=[
                          {"role": role, "content":message}
                           for role, message, _ in db_rows
                         
                     ]

            else:
                st.info("No applicants found. Please create an applicant first.")

                
                 
        
            # # st.warning("Please save applicant details first.")
            # if selected_id:
            #     st.session_state.applicant_id= selected_id
            #     db_rows=get_chat_history_by_applicant(selected_id)
            #     st.session_state.chat_history=[
            #         {"role": role, "content":message}
            #         for role, message, _ in db_rows
            #     ]
        else:
            chat_rows= get_chat_history_by_applicant(st.session_state.applicant_id)

            st.markdown(f"### Chat history for Applicant ID: {st.session_state.applicant_id}")
 
        if not chat_rows:
            st.info("No chat history found.")
        else:
            # st.markdown("Previous chat History:")
 
            for role,message, timestamp in chat_rows:
                with st.chat_message(role):
                    st.caption(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
                    st.write(message)
 
               
    st.markdown("### Your Loan Companion")
    chat_container = st.container(height=400, border=True)
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
   
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message...", label_visibility="collapsed")
        send_button = st.form_submit_button("Send", use_container_width=True, type="primary")
   
    if send_button and user_input:
        with st.spinner(" Retrieving FAQs and generating response..."):
            form_context = get_form_context(
                first_name, last_name, dob, gender, marital_status,
                phone, email, aadhaar, pan, current_address, permanent_address
            )
           
            bot_response = generate_response(user_input, form_context)
       
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        #change made
        save_chat_message(st.session_state.applicant_id,st.session_state.session_id, "user", user_input)
 
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
       
        save_chat_message(st.session_state.applicant_id,st.session_state.session_id, "assistant", bot_response)
 
 
       
        st.rerun()

######

import streamlit as st
from datetime import date
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
import pyttsx3
import numpy as np
import queue
import sounddevice as sd
import os
from dotenv import load_dotenv
import uuid
import psycopg2
from vosk import Model, KaldiRecognizer
# import bitsandbytes

load_dotenv()

st.set_page_config(page_title="Loan Application with RAG", layout="wide")


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT"),
    )


def save_applicant_data(
    first_name,
    middle_name,
    last_name,
    dob,
    gender,
    marital_status,
    phone,
    email,
    aadhaar,
    pan,
    current_address,
    permanent_address,
):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO loan_applicants (
                first_name, middle_name, last_name, dob, gender, marital_status,
                phone, email, aadhaar, pan, current_address, permanent_address
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        values = (
            first_name,
            middle_name,
            last_name,
            dob,
            gender,
            marital_status,
            phone,
            email,
            aadhaar,
            pan,
            current_address,
            permanent_address,
        )
        cursor.execute(query, values)
        applicant_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        return True, applicant_id
    except Exception as e:
        return False, None


def save_chat_message(session_id, role, message):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO chat_history(
            session_id,
            role,
            message
            )
            VALUES(%s,%s,%s)
            """,
            (session_id, role, message),
        )
        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        st.error(f"Failed to save chat message: {e}")


def get_chat_history_from_db(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, message, created_at
        FROM chat_history
        WHERE session_id = %s
        ORDER BY created_at
        """,
        (session_id,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return rows


def get_chat_history_by_applicant(applicant_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, message, created_at
        FROM chat_history
        WHERE applicant_id = %s
        ORDER BY created_at
        """,
        (applicant_id,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return rows


##############################################################


@st.cache_resource
def load_model():
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    model_path = r"gemma"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    return model, tokenizer


@st.cache_resource
def load_faq_system():
    embedding_model = SentenceTransformer(r"all-MiniLM-L6-v2")
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection("loan_faqs")

    try:
        with open("faqs_data.json", "r", encoding="utf-8") as f:
            faq_data = json.load(f)
    except FileNotFoundError:
        st.error("faqs_data.json not found!")
        return embedding_model, collection

    if collection.count() == 0:
        for faq in faq_data.get("faqs", []):
            text = f"Q: {faq['question']}\nA: {faq['answer']}"
            embedding = embedding_model.encode(text).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[
                    {
                        "question": faq["question"],
                        "category": faq.get("category", ""),
                        "id": faq["id"],
                    }
                ],
                ids=[faq["id"]],
            )
        st.success(f"Loaded {len(faq_data.get('faqs', []))} FAQs into vector database")

    return embedding_model, collection


def get_form_context(
    first_name,
    last_name,
    dob,
    gender,
    marital_status,
    phone,
    email,
    aadhaar,
    pan,
    current_address,
    permanent_address,
    occupation_type,
    annual_income,
    employer_name,
    designation,
    work_experience,
    work_location,
    office_address,
    loan_type,
    loan_amount,
    loan_duration,
    emi_date,
    existing_loans,
    nominee_name,
    nominee_dob,
    nominee_relationship,
    nominee_mobile,
    nominee_email,
    nominee_pan,
    nominee_address,
):
    context_parts = []

    # Personal Info
    if first_name:
        context_parts.append(f"Applicant's first name: {first_name}")
    if last_name:
        context_parts.append(f"Applicant's last name: {last_name}")

    if dob:
        age = (date.today() - dob).days // 365
        context_parts.append(f"Applicant's date of birth: {dob} (Age: {age} years)")

    if gender and gender != "Select Gender":
        context_parts.append(f"Gender: {gender}")

    if marital_status and marital_status != "Select Marital Status":
        context_parts.append(f"Marital status: {marital_status}")

    if phone:
        context_parts.append(f"Phone number provided: Yes")

    if email:
        context_parts.append(f"Email provided: Yes")

    if aadhaar:
        context_parts.append(f"Aadhaar card number provided: Yes")

    if pan:
        context_parts.append(f"PAN card number provided: Yes")

    if current_address:
        context_parts.append(f"Current address provided: Yes")

    if permanent_address:
        context_parts.append(f"Permanent address provided: Yes")

    # Work Information
    if occupation_type and occupation_type != "Employed":
        context_parts.append(f"Occupation: {occupation_type}")

    if annual_income:
        context_parts.append(f"Annual income: ₹{annual_income}")

    if employer_name:
        context_parts.append(f"Employer: {employer_name}")

    if designation:
        context_parts.append(f"Designation: {designation}")

    if work_experience:
        context_parts.append(f"Work experience: {work_experience} years")

    if work_location and work_location != "Select a City":
        context_parts.append(f"Work location: {work_location}")

    if office_address:
        context_parts.append(f"Office address provided: Yes")

    # Loan Request Information
    if loan_type and loan_type != "Select Loan Type":
        context_parts.append(f"Requested loan type: {loan_type}")

    if loan_amount:
        context_parts.append(f"Requested loan amount: ${loan_amount}")

    if loan_duration:
        context_parts.append(f"Loan duration: {loan_duration} years")

    if emi_date and emi_date != "Select a date":
        context_parts.append(f"Preferred EMI date: {emi_date}")

    if existing_loans:
        context_parts.append(f"Has existing loans: {existing_loans}")

    # Nominee Information
    if nominee_name:
        context_parts.append(f"Nominee name: {nominee_name}")

    if nominee_dob:
        nominee_age = (date.today() - nominee_dob).days // 365
        context_parts.append(f"Nominee age: {nominee_age} years")

    if nominee_relationship:
        context_parts.append(f"Nominee relationship: {nominee_relationship}")

    if nominee_mobile:
        context_parts.append(f"Nominee contact provided: Yes")

    if nominee_email:
        context_parts.append(f"Nominee email provided: Yes")

    if nominee_pan:
        context_parts.append(f"Nominee PAN provided: Yes")

    if nominee_address:
        context_parts.append(f"Nominee address provided: Yes")

    missing = []
    if not first_name:
        missing.append("First name")
    if not last_name:
        missing.append("Last name")
    if not dob:
        missing.append("Date of birth")
    if not aadhaar:
        missing.append("Aadhaar card")
    if not pan:
        missing.append("PAN card")

    if missing:
        context_parts.append(f"Missing required fields: {', '.join(missing)}")

    return "\n".join(context_parts) if context_parts else "No form data filled yet."


##############################
def check_recent_context(chat_history, last_n=3):
    if not chat_history:
        return False

    recent_messages = chat_history[-last_n:]

    loan_indicators = {
        "strong": [
            "loan",
            "emi",
            "mortgage",
            "eligibility",
            "application",
            "apply",
            "disbursement",
            "approval",
        ],
        "moderate": [
            "bank",
            "finance",
            "income",
            "credit",
            "interest",
            "document",
            "aadhaar",
            "pan",
            "salary",
        ],
        "weak": ["money", "pay", "amount", "need", "want", "help", "form"],
    }

    score = 0.0

    for idx, msg in enumerate(recent_messages):
        content = msg["content"].lower()
        recency_weight = 1.0 - (0.3 * (len(recent_messages) - idx - 1))

        if any(kw in content for kw in loan_indicators["strong"]):
            score += 1.5 * recency_weight
        elif any(kw in content for kw in loan_indicators["moderate"]):
            score += 1.0 * recency_weight
        elif any(kw in content for kw in loan_indicators["weak"]):
            score += 0.3 * recency_weight

    return score >= 1.0

def is_loan_banking_query(user_input, embedding_model):
    user_input_lower = user_input.lower().strip()
    
    greetings_and_common = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'thanks', 'thank you', 'ok', 'okay', 'yes', 'no', 'bye', 'goodbye',
        'help', 'start', 'begin', 'continue', 'what can you do', 'who are you'
    ]
    
    is_greeting = any(user_input_lower == greeting or user_input_lower.startswith(greeting + ' ') or user_input_lower.startswith(greeting + '!') for greeting in greetings_and_common)
    
    if is_greeting:
        return True, True
    
    loan_banking_keywords = [
        'loan', 'bank', 'finance', 'credit', 'interest', 'emi', 'repayment',
        'mortgage', 'eligibility', 'application', 'form', 'document', 'income',
        'salary', 'employment', 'aadhaar', 'pan', 'kyc', 'approval', 'disbursement',
        'tenure', 'amount', 'personal loan', 'home loan', 'car loan', 'education loan',
        'business loan', 'collateral', 'guarantor', 'nominee', 'processing fee',
        'down payment', 'rate', 'scheme', 'insurance', 'financial', 'money', 'pay',
        'debt', 'borrow', 'lend', 'apply', 'account', 'balance', 'deposit'
    ]
    
    if any(keyword in user_input_lower for keyword in loan_banking_keywords):
        return True, False
    return False, False


def should_respond(user_input, chat_history, embedding_model, faq_collection):

    is_relevant, is_greeting = is_loan_banking_query(user_input, embedding_model)
    if is_greeting:
        return True, "greeting", None

    if is_relevant:
        return True, "clear_relevant", None

    explicit_offtopic = [
        "weather",
        "temperature",
        "rain",
        "sunny",
        "climate",
        "recipe",
        "cook",
        "food",
        "restaurant",
        "meal",
        "dish",
        "movie",
        "film",
        "cinema",
        "actor",
        "actress",
        "sport",
        "football",
        "cricket",
        "basketball",
        "game",
        "match",
        "music",
        "song",
        "singer",
        "album",
        "politics",
        "election",
        "government",
        "minister",
        "celebrity",
        "famous person",
        "star",
    ]

    user_input_lower = user_input.lower()
    if any(word in user_input_lower for word in explicit_offtopic):
        return (
            False,
            "hard_reject",
            "I apologize, but I can only assist with loan applications, banking, and financial matters. Your query seems irrelevent, if you feel I am wrong, please rephrase the query. How can I help you with your loan application?",
        )

    recent_was_relevant = check_recent_context(chat_history, last_n=3)
    if recent_was_relevant:
        return True, "contextual_relevant", None

    faq_context, distances = retrieve_relevant_faqs(
        user_input, embedding_model, faq_collection, top_k=3
    )
    if distances and len(distances) > 0 and min(distances) < 0.7:
        return True, "rag_relevant", None

    return (
        False,
        "soft_reject",
        "I'm here to help with your loan application. Are you asking about something related to loans, banking, or your application? Please let me know how I can assist you with your loan needs.",
    )


##############################


def retrieve_relevant_faqs(user_input, embedding_model, faq_collection, top_k=3):
    try:
        query_embedding = embedding_model.encode(user_input).tolist()

        results = faq_collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )

        if results["documents"] and results["documents"][0]:
            return "\n\n".join(results["documents"][0]), (
                results["distances"][0] if "distances" in results else None
            )
        return "", None
    except Exception as e:
        st.error(f"Error retrieving FAQs: {str(e)}")
        return "", None


def text_to_speech(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        engine.setProperty("volume", 0.9)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.warning(f"TTS error: {e}")


def generate_response(user_input, form_context):
    try:
        model, tokenizer = load_model()
        embedding_model, faq_collection = load_faq_system()

        faq_context, distances = retrieve_relevant_faqs(
            user_input, embedding_model, faq_collection, top_k=3
        )
        system_prompt = f"""You are a specialized loan application assistant. Your ONLY purpose is to help users with loan applications, banking, and financial matters.

APPLICANT'S CURRENT FORM STATUS:
{form_context}

RELEVANT FAQ INFORMATION:
{faq_context}

STRICT GUIDELINES:
- ONLY answer questions about: loans, banking, finance, this application form, eligibility, documents, and related topics
- If asked about ANYTHING else (weather, recipes, sports, general knowledge, etc.), politely decline and redirect to loan-related topics
- Use the FAQ information to provide accurate answers
- Consider the applicant's form progress for personalized guidance
- If they ask about eligibility, use their age and provided details
- If asked what's needed, mention what they've provided and what's missing
- Be as concise as possible, helpful, and accurate
- No Emojis and maitain a professional tone
- NEVER answer off-topic questions, even if you know the answer"""

        messages = [
            {"role": "user", "content": system_prompt},
            {
                "role": "assistant",
                "content": "I'll provide personalized assistance based on the applicant's information and FAQ knowledge.",
            },
        ]

        for msg in st.session_state.chat_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_input})

        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = input_ids.to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=500,
            temperature=0.3,
            do_sample=True,
            top_p=0.8,
            repetition_penalty=1.1,
        )

        generated_ids = outputs[0][input_ids.shape[-1] :]
        response = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        response = response.replace("<end_of_turn>", "").strip()

        return (
            response if response else "I'm here to help! What would you like to know?"
        )
    except Exception as e:
        return f"Error generating response: {str(e)}"


########################################
MODEL_PATH = "C:\\Users\\2031063\\AI Bootcamp\\FAQ CHATBOT PROJECT\\model\\vosk-model-small-en-us-0.15"

q = queue.Queue()
model = Model(MODEL_PATH)
samplerate = 16000


def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))


def offline_speech_to_text(duration=5):
    recognizer = KaldiRecognizer(model, samplerate)

    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        for _ in range(int(duration * samplerate / 8000)):
            data = q.get()
            recognizer.AcceptWaveform(data)

    result = json.loads(recognizer.FinalResult())
    return result.get("text", "")


########################################


left_col, right_col = st.columns([2, 1], gap="large")

with left_col:
    # st.write(st.session_state)
    st.title("Loan Application")
    st.markdown("---")

    form_container = st.container(height=650, border=False)
    with form_container:
        st.subheader("Personal Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            first_name = st.text_input("First Name", placeholder="John")
        with col2:
            middle_name = st.text_input("Middle Name (Optional)", placeholder="Smith")
        with col3:
            last_name = st.text_input("Last Name", placeholder="Doe")

        col1, col2, col3 = st.columns(3)
        with col1:
            min_date = date.today().replace(year=date.today().year - 100)
            dob = st.date_input(
                "Date of Birth", value=None, min_value=min_date, max_value=date.today()
            )
        with col2:
            gender = st.selectbox(
                "Gender",
                ["Select Gender", "Male", "Female", "Other", "Prefer not to say"],
            )
        with col3:
            marital_status = st.selectbox(
                "Marital Status",
                ["Select Marital Status", "Single", "Married", "Divorced", "Widowed"],
            )

        col1, col2 = st.columns(2)
        with col1:
            phone = st.text_input("Phone Number", placeholder="12345 67890")
        with col2:
            email = st.text_input("Email Address", placeholder="john.doe@example.com")

        col1, col2 = st.columns(2)
        with col1:
            aadhaar = st.text_input("Aadhaar Card Number", placeholder="XXXX XXXX XXXX")
        with col2:
            pan = st.text_input("PAN Card Number", placeholder="ABCDE1234F")

        st.markdown("")
        current_address = st.text_area(
            "Current Address",
            placeholder="Enter your current address",
            height="stretch",
        )

        same_address = st.checkbox("Permanent Address is the same as Current Address")

        if same_address:
            permanent_address = current_address
            st.text_area(
                "Permanent Address",
                value=current_address,
                height="stretch",
                disabled=True,
            )
        else:
            permanent_address = st.text_area(
                "Permanent Address",
                placeholder="Enter your permanent address",
                height="stretch",
            )

        st.markdown("---")
        st.subheader("Work Information")

        col1, col2 = st.columns(2)
        with col1:
            occupation_type = st.selectbox(
                "Occupation Type",
                ["Employed", "Self-Employed", "Student", "Retired", "Unemployed"],
            )
        with col2:
            annual_income = st.text_input(
                "Annual Income (₹)", placeholder="e.g., 50000"
            )

        col1, col2 = st.columns(2)
        with col1:
            employer_name = st.text_input("Employer Name", placeholder="e.g., ABC Corp")
        with col2:
            designation = st.text_input(
                "Designation", placeholder="e.g., Software Engineer"
            )

        col1, col2 = st.columns(2)
        with col1:
            work_experience = st.text_input(
                "Work Experience (in years)", placeholder="e.g., 5"
            )
        with col2:
            work_location = st.selectbox(
                "Location",
                [
                    "Select a City",
                    "Mumbai",
                    "Delhi",
                    "Bangalore",
                    "Hyderabad",
                    "Chennai",
                    "Kolkata",
                    "Pune",
                    "Ahmedabad",
                    "Other",
                ],
            )

        office_address = st.text_area(
            "Office Address", placeholder="Office Address", height=100
        )

        st.markdown("---")
        st.subheader("Your Loan Request")

        col1, col2 = st.columns(2)
        with col1:
            loan_type = st.selectbox(
                "Loan Type",
                [
                    "Select Loan Type",
                    "Personal Loan",
                    "Home Loan",
                    "Car Loan",
                    "Education Loan",
                    "Business Loan",
                ],
            )
        with col2:
            loan_amount = st.text_input(
                "Required Loan Amount ($)", placeholder="e.g., 500000"
            )

        col1, col2 = st.columns(2)
        with col1:
            loan_duration = st.text_input(
                "Loan Duration (in years)", placeholder="e.g., 10"
            )
        with col2:
            emi_date = st.selectbox(
                "Preferred EMI Date", ["Select a date"] + [str(i) for i in range(1, 29)]
            )

        st.markdown("Do you have any existing loans?")
        existing_loans = st.radio(
            "", ["Yes", "No"], horizontal=True, label_visibility="collapsed"
        )

        st.markdown("---")
        st.subheader("Nominee Information")

        col1, col2 = st.columns(2)
        with col1:
            nominee_name = st.text_input("Nominee Name", placeholder="Nominee Name")
        with col2:
            nominee_dob = st.date_input(
                "Nominee DOB",
                value=None,
                min_value=date.today().replace(year=date.today().year - 100),
                max_value=date.today(),
                format="MM/DD/YYYY",
            )

        col1, col2 = st.columns(2)
        with col1:
            nominee_relationship = st.text_input(
                "Relationship to Applicant", placeholder="e.g., Spouse, Parent"
            )
        with col2:
            nominee_mobile = st.text_input(
                "Nominee Mobile Number", placeholder="e.g., (123) 456-7890"
            )

        col1, col2 = st.columns(2)
        with col1:
            nominee_email = st.text_input(
                "Nominee Email ID", placeholder="e.g., nominee@example.com"
            )
        with col2:
            nominee_pan = st.text_input(
                "Nominee PAN Number", placeholder="e.g., ABCDE1234F"
            )

        nominee_address = st.text_area(
            "Nominee Address", placeholder="Nominee Address", height=100
        )

    st.markdown("")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Save Details", use_container_width=True):
            success, applicant_id = save_applicant_data(
                first_name,
                middle_name,
                last_name,
                dob,
                gender,
                marital_status,
                phone,
                email,
                aadhaar,
                pan,
                current_address,
                permanent_address,
            )

            if success:
                st.session_state.applicant_id = applicant_id

                db_rows = get_chat_history_by_applicant(applicant_id)
                st.session_state.chat_history = [
                    {"role": role, "content": message} for role, message, _ in db_rows
                ]
                st.success(f"Applicant ID: {applicant_id}")
            else:
                st.error("Rejected!")

session_dict = {}

with right_col:

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT session_id FROM chat_history")
    session_ids = [row[0] for row in cursor.fetchall()]
    for index, value in enumerate(session_ids):
        session_dict[value] = "Session " + str(index)
    cursor.close()
    conn.close()

    def format_display_name(option_value):
        return session_dict[option_value]

    st.markdown("##### Past Sessions")
    if session_ids:
        selected_id = st.selectbox(
            "Select Session ID",
            session_ids,
            index=None,
            key="session_selector",
            format_func=format_display_name,
        )

        if (
            "session_ids" not in st.session_state
            or st.session_state.session_ids != selected_id
        ) and selected_id != None:
            st.session_state.session_id = selected_id
            db_rows = get_chat_history_from_db(selected_id)
            st.session_state.chat_history = [
                {"role": role, "content": message} for role, message, _ in db_rows
            ]
    else:
        st.info("No sessions found")

    #####################################

    st.markdown("### Your Loan Companion")
    st.info(
        """
    Hi there! I'm here to help you with your loan application.
    Feel free to ask me any questions!
    """
    )

    chat_container = st.container(height=400, border=True)
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Read Last Response", use_container_width=True):
            if st.session_state.chat_history:
                last_msg = [
                    m for m in st.session_state.chat_history if m["role"] == "assistant"
                ]
                if last_msg:
                    with st.spinner("Speaking..."):
                        text_to_speech(last_msg[-1]["content"])
    with col2:
        if st.button("Speak", use_container_width=True):
            with st.spinner("Listening..."):
                spoken_text = offline_speech_to_text()

            if spoken_text:
                st.session_state.chat_input = spoken_text
                st.toast("🎙 Speech captured")
                st.rerun()

    #####################################
    if st.session_state.voice_text:
        st.session_state.chat_input = st.session_state.voice_text
        st.session_state.voice_chat = ""

    with st.form(key="chat_form"):
        user_input = st.text_input(
            "Type your message...", label_visibility="collapsed", key="chat_input"
        )

        send_button = st.form_submit_button(
            "Send", type="primary", use_container_width=True
        )

    if send_button and st.session_state.chat_input:
        user_input = st.session_state.chat_input
        
        #####################
        with st.spinner("Analyzing your question......."):
            embedding_model, faq_collection = load_faq_system()
            can_respond, response_type, rejection_message = should_respond(
                user_input,
                st.session_state.chat_history,
                embedding_model,
                faq_collection
            )
            
            if not can_respond:
                bot_response = rejection_message
                
        #####################
        
        if can_respond:
            with st.spinner("Retrieving FAQs and generating response..."):
                form_context = get_form_context(
                    first_name,
                    last_name,
                    dob,
                    gender,
                    marital_status,
                    phone,
                    email,
                    aadhaar,
                    pan,
                    current_address,
                    permanent_address,
                    occupation_type,
                    annual_income,
                    employer_name,
                    designation,
                    work_experience,
                    work_location,
                    office_address,
                    loan_type,
                    loan_amount,
                    loan_duration,
                    emi_date,
                    existing_loans,
                    nominee_name,
                    nominee_dob,
                    nominee_relationship,
                    nominee_mobile,
                    nominee_email,
                    nominee_pan,
                    nominee_address,
                )

                bot_response = generate_response(user_input, form_context)

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        ###################
        save_chat_message(st.session_state.session_id, "user", user_input)
        ###################

        st.session_state.chat_history.append(
            {"role": "assistant", "content": bot_response}
        )

        #############
        save_chat_message(st.session_state.session_id, "assistant", bot_response)

        st.rerun()

 
