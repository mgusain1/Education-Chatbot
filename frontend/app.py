import streamlit as st
import requests

st.set_page_config(page_title="Uni Chatbot", layout="centered")
st.title("Ask about any University")

if "search_results" not in st.session_state:
    st.session_state.search_results = []
    
if "search_query" not in st.session_state:
    st.session_state.search_query = []

query = st.text_input("Ask anything about US university")
if st.button("Ask"):
    if query.strip()=="":
        st.warning("Please enter a question")
    else:
        with st.spinner("Searching..."):
            response = requests.post(
                "http://localhost:8000/ask",
                json = {"query":query}
            )
            if response.status_code==200:
                results = response.json()["matches"]
                st.session_state.search_results = results
                st.session_state.search_query = query
            else:
                st.error("Failed to get results")
        
if st.session_state.search_results:
    for idx, uni in enumerate(st.session_state.search_results, 1):
        st.subheader(f"{idx}. {uni['name']} ({uni['city']}, {uni['state']})")
        st.write(f"**Type:** {uni['control_type']}")
        st.write(f"In-State Tuition: ${uni['tuition_in_state']}")
        st.write(f"Out-of-State Tuition: ${uni['tuition_out_state']}")
        st.write(f"Undergrad: {uni['undergrad']}, Grad: {uni['grad']}")
        url = uni["website"]
        if not url.startswith("http"):
            url = "https://" + url
        st.markdown(f"[Website]({url})")

        if st.button(f"Show me Admission requirements for {uni['name']}", key=f"req_{idx}"):
            with st.spinner("Fetching University Requirements..."):
                req_response = requests.post(
                    "http://localhost:8000/admission-requirements",
                    json={"university": uni["name"]}
                )
                if req_response.status_code == 200:
                    requirements = req_response.json()['requirements']
                    with st.expander("📋 Admission Requirements"):
                        st.markdown(requirements)
                else:
                    st.error("Failed to fetch requirements")
        st.markdown("---")
