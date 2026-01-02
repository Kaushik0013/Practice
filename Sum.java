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

# -------------------- ENV & SESSION --------------------
load_dotenv()

st.set_page_config(page_title="Loan Application with RAG", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# -------------------- DB CONNECTION --------------------
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
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO loan_applicants (
            first_name, middle_name, last_name, dob, gender, marital_status,
            phone, email, aadhaar, pan, current_address, permanent_address
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        first_name, middle_name, last_name, dob, gender, marital_status,
        phone, email, aadhaar, pan, current_address, permanent_address
    ))
    conn.commit()
    cursor.close()
    conn.close()

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
    return rows

# -------------------- LOAD CHAT HISTORY (PERSISTENT) --------------------
if "chat_history" not in st.session_state:
    rows = load_chat_history(st.session_state.session_id)
    st.session_state.chat_history = [
        {"role": role, "content": msg} for role, msg in rows
    ]

# -------------------- MODEL & RAG --------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gemma", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "gemma",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

@st.cache_resource
def load_faq_system():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection("loan_faqs")

    if collection.count() == 0:
        with open("card_activation_faqs.json") as f:
            data = json.load(f)
        for faq in data["faqs"]:
            text = f"Q: {faq['question']}\nA: {faq['answer']}"
            collection.add(
                documents=[text],
                embeddings=[embedder.encode(text).tolist()],
                ids=[faq["id"]]
            )
    return embedder, collection

def generate_response(user_input, context):
    model, tokenizer = load_model()
    embedder, collection = load_faq_system()

    faq = collection.query(
        query_embeddings=[embedder.encode(user_input).tolist()],
        n_results=3
    )
    faq_text = "\n".join(faq["documents"][0]) if faq["documents"] else ""

    messages = [
        {"role": "system", "content": f"{context}\n\nFAQ:\n{faq_text}"},
        *st.session_state.chat_history[-10:],
        {"role": "user", "content": user_input}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    output = model.generate(input_ids, max_new_tokens=400)
    response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return response.strip()

# -------------------- UI --------------------
left_col, right_col = st.columns([2, 1], gap="large")

# -------- LEFT: PERSONAL DETAILS (UNCHANGED) --------
with left_col:
    st.title("Personal Details")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    first_name = col1.text_input("First Name")
    middle_name = col2.text_input("Middle Name")
    last_name = col3.text_input("Last Name")

    col1, col2, col3 = st.columns(3)
    dob = col1.date_input("Date of Birth", value=None, max_value=date.today())
    gender = col2.selectbox("Gender", ["Select Gender", "Male", "Female"])
    marital_status = col3.selectbox("Marital Status", ["Select", "Single", "Married"])

    phone = st.text_input("Phone")
    email = st.text_input("Email")
    aadhaar = st.text_input("Aadhaar")
    pan = st.text_input("PAN")

    current_address = st.text_area("Current Address")
    permanent_address = st.text_area("Permanent Address")

    if st.button("Save Details"):
        save_applicant_data(
            first_name, middle_name, last_name, dob, gender,
            marital_status, phone, email, aadhaar, pan,
            current_address, permanent_address
        )
        st.success("Details saved successfully!")

# -------- RIGHT: CHATBOT --------
with right_col:
    st.markdown("### 🤖 Your Loan Companion")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask something...")

    if user_input:
        context = f"Applicant name: {first_name} {last_name}"
        bot_response = generate_response(user_input, context)

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        save_chat_message(st.session_state.session_id, "user", user_input)

        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        save_chat_message(st.session_state.session_id, "assistant", bot_response)

        st.rerun()
