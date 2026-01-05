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
 
