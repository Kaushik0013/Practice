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

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ---------------- DB CONNECTION ----------------
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )

# ---------------- SAVE APPLICANT (CHANGED) ----------------
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
            RETURNING id;
        """

        cursor.execute(query, (
            first_name, middle_name, last_name, dob, gender, marital_status,
            phone, email, aadhaar, pan, current_address, permanent_address
        ))

        applicant_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()

        return True, applicant_id

    except Exception as e:
        return False, str(e)

# ---------------- CHAT HISTORY (CHANGED) ----------------
def save_chat_message(applicant_id, role, message):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO chat_history (applicant_id, role, message)
            VALUES (%s, %s, %s)
            """,
            (applicant_id, role, message)
        )

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        st.error(f"Failed to save chat message: {e}")

def get_chat_history_from_db(applicant_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, message, created_at
        FROM chat_history
        WHERE applicant_id = %s
        ORDER BY created_at
        """,
        (applicant_id,)
    )

    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# ---------------- MODEL / FAQ ----------------
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

    with open("card_activation_faqs.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    if collection.count() == 0:
        for faq in faq_data["faqs"]:
            text = f"Q: {faq['question']}\nA: {faq['answer']}"
            embedding = embedding_model.encode(text).tolist()
            collection.add(
                embeddings=[embedding],
                documents=[text],
                ids=[faq["id"]]
            )

    return embedding_model, collection

# ---------------- UI ----------------
left_col, right_col = st.columns([2, 1], gap="large")

# -------- LEFT COLUMN (UNCHANGED) --------
with left_col:
    st.title("Personal Details")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    first_name = col1.text_input("First Name")
    middle_name = col2.text_input("Middle Name")
    last_name = col3.text_input("Last Name")

    col1, col2, col3 = st.columns(3)
    dob = col1.date_input("Date of Birth", value=None)
    gender = col2.selectbox("Gender", ["Select Gender", "Male", "Female"])
    marital_status = col3.selectbox("Marital Status", ["Select", "Single", "Married"])

    phone = st.text_input("Phone")
    email = st.text_input("Email")
    aadhaar = st.text_input("Aadhaar")
    pan = st.text_input("PAN")

    current_address = st.text_area("Current Address")
    permanent_address = st.text_area("Permanent Address")

    if st.button("Save Details"):
        success, result = save_applicant_data(
            first_name, middle_name, last_name, dob, gender, marital_status,
            phone, email, aadhaar, pan, current_address, permanent_address
        )

        if success:
            st.session_state.applicant_id = result
            st.success("Details saved successfully!")
        else:
            st.error(result)

# -------- RIGHT COLUMN (MINIMAL CHANGE) --------
with right_col:

    if st.button("View Chat History"):
        if "applicant_id" not in st.session_state:
            st.warning("Please save applicant details first.")
        else:
            rows = get_chat_history_from_db(st.session_state.applicant_id)
            for role, message, ts in rows:
                with st.chat_message(role):
                    st.caption(ts.strftime("%Y-%m-%d %H:%M:%S"))
                    st.write(message)

    st.markdown("### Your Loan Companion")

    chat_container = st.container(height=400, border=True)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message...")
        send = st.form_submit_button("Send")

    if send and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if "applicant_id" in st.session_state:
            save_chat_message(st.session_state.applicant_id, "user", user_input)

        bot_response = "Thanks for your message! (LLM response here)"
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

        if "applicant_id" in st.session_state:
            save_chat_message(st.session_state.applicant_id, "assistant", bot_response)

        st.rerun()

