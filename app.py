import streamlit as st
from streamlit_chat import message

from constants import (
    APP_NAME,
    CHUNK_SIZE,
    FETCH_K,
    MAX_TOKENS,
    PAGE_ICON,
    PAGE_DESC,
    TEMPERATURE,
    K,
)
from utils import (
    authenticate,
    delete_uploaded_file,
    generate_response,
    logger,
    save_uploaded_file,
    update_chain,
)


st.set_page_config(
    page_title=APP_NAME, page_icon=PAGE_ICON
)
st.markdown(
    f"<h1 style='text-align: center;'>{APP_NAME} {PAGE_ICON} <br> {PAGE_DESC} </h1>",
    unsafe_allow_html=True,
)

# set all the session states for chat history and file
if "past" not in st.session_state:
    st.session_state["past"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "data_source" not in st.session_state:
    st.session_state["data_source"] = None
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

# set all the session states for authentication
if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None

# set all the session states LLM hyperparams
if "k" not in st.session_state:
    st.session_state["k"] = K
if "fetch_k" not in st.session_state:
    st.session_state["fetch_k"] = FETCH_K
if "chunk_size" not in st.session_state:
    st.session_state["chunk_size"] = CHUNK_SIZE
if "temperature" not in st.session_state:
    st.session_state["temperature"] = TEMPERATURE
if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = MAX_TOKENS

# no sidebar, show authentication on single page and hide it after success
if not st.session_state["auth_ok"]:
    with st.form("authentication"):
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="This field is mandatory",
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            authenticate(openai_api_key)
            st.experimental_rerun()

# After authentication is good to go
# Show file upload and data source inputs
# toggle upload options so not to show both of them after when user used one of them
uploaded_file = None
data_source = None
if st.session_state["data_source"] is None and st.session_state["auth_ok"]:
    # hide 
    uploaded_file = st.file_uploader("Upload a file")
    st.write("OR")
    data_source = st.text_input(
        "Point to any web url",
        placeholder="Any web url pointing to a file",
    )

# Take user input for web link and create data source
if data_source and data_source != st.session_state["data_source"] and st.session_state["auth_ok"]:
    logger.info(f"web link provided: '{data_source}'")
    st.session_state["data_source"] = data_source
    update_chain()

# Take uploaded file and create data source
if uploaded_file and uploaded_file != st.session_state["uploaded_file"]:
    logger.info(f"Uploaded file provided: '{uploaded_file.name}'")
    st.session_state["uploaded_file"] = uploaded_file
    data_source = save_uploaded_file(uploaded_file)
    st.session_state["data_source"] = data_source
    update_chain()
    delete_uploaded_file(uploaded_file)


# response container
response_container = st.container()
# user intput container
container = st.container()

# reload the whole container
if st.session_state["uploaded_file"]:
    st.write(f"You are Chatting with 🗄️: {st.session_state['uploaded_file'].name}")

if st.session_state["auth_ok"]:
    with container:
        with st.form(key="prompt_input", clear_on_submit=True):
            user_input = st.text_area("You:", key="input", height=100)
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            output = generate_response(user_input)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    # clear conversation ui
    clear_button = st.button(label="Clear Conversation")

    if st.session_state["generated"]:
        with response_container:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))
    
    # clear conversation logic
    if clear_button:
        st.session_state["past"] = []
        st.session_state["generated"] = []
        st.session_state["chat_history"] = []
        st.session_state["data_source"] = None
        st.session_state["uploaded_file"] = None
        uploaded_file = None
        data_source = None
        st.experimental_rerun()
