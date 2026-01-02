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

# ===============================
# Session identity (persistent)
# ===============================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ===============================
# DB Connection
# ===============================
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )

# ===============================
# Chat persistence
# ===============================
def save_chat_message(session_id, role, message):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_history (session_id, role, message)
        VALUES (%s, %s, %s)
        """,
        (session_id, role, message)
    )
    conn.commit()
    cur.close()
    conn.close()

def load_chat_history(session_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, message
        FROM chat_history
        WHERE session_id = %s
        ORDER BY created_at
        """,
        (session_id,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# ===============================
# 🔧 LOAD DB CHAT INTO SESSION
# ===============================
if "chat_history" not in st.session_state:
    db_rows = load_chat_history(st.session_state.session_id)
    st.session_state.chat_history = [
        {"role": r, "content": m} for r, m in db_rows
    ]

# ===============================
# Model + RAG (UNCHANGED)
# ===============================
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

    with open("card_activation_faqs.json", "r") as f:
        faq_data = json.load(f)

    if collection.count() == 0:
        for faq in faq_data["faqs"]:
            text = f"Q: {faq['question']}\nA: {faq['answer']}"
            collection.add(
                documents=[text],
                embeddings=[embedding_model.encode(text).tolist()],
                ids=[faq["id"]]
            )

    return embedding_model, collection

def generate_response(user_input, form_context):
    model, tokenizer = load_model()
    embed_model, faq_collection = load_faq_system()

    messages = [
        {"role": "system", "content": form_context}
    ]

    messages.extend(st.session_state.chat_history[-10:])
    messages.append({"role": "user", "content": user_input})

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    )

    return response.strip()

# ===============================
# UI (UNCHANGED)
# ===============================
left_col, right_col = st.columns([2, 1], gap="large")

with right_col:
    st.markdown("### 💬 Your Loan Companion")

    chat_container = st.container(height=400, border=True)
    with chat_container:
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message...")
        send = st.form_submit_button("Send")

    if send and user_input:
        response = generate_response(user_input, "Loan application assistant")

        # 🔧 Save to DB
        save_chat_message(st.session_state.session_id, "user", user_input)
        save_chat_message(st.session_state.session_id, "assistant", response)

        # 🔧 Update session
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )

        st.rerun()
