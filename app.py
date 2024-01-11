import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""                           #variabel teks untuk menyimpan teks yang dihasilkan nantinya 
    for pdf in pdf_docs:                #Melakukan iterasi melalui setiap file PDF dalam daftar 
        pdf_reader = PdfReader(pdf)     #Membuat objek pembaca PDF (PdfReader) untuk membaca konten dari file PDF yang sedang diproses.
        for page in pdf_reader.pages:   #Melakukan iterasi melalui setiap halaman dalam file PDF.
            text += page.extract_text() #Mengambil teks dari setiap halaman dan menambahkannya ke variabel text
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(  #Membuat objek text_splitter dari kelas CharacterTextSplitter
        separator="\n",                     #Separator yang digunakan untuk membagi teks menjadi potongan-potongan.
        chunk_size=1000,                    
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text) #Memanggil metode split_text dari objek text_splitter untuk membagi teks menjadi potongan-potongan. Potongan-potongan ini disimpan dalam variabel chunks.
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings) #embuat objek vectorstore menggunakan FAISS, yang merupakan sebuah library untuk pencarian vektor yang efisien
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(                           #Membuat objek memory dari kelas ConversationBufferMemory. Kelas ini mungkin bertanggung jawab untuk menyimpan dan mengelola sejarah percakapan atau memori percakapan
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm( #embuat objek conversation_chain dari kelas ConversationalRetrievalChain. Kelas ini mungkin bertanggung jawab untuk mengelola alur percakapan, pengambilan informasi, dan interaksi dengan model bahasa
        llm=llm,
        retriever=vectorstore.as_retriever(),                   #Objek yang berfungsi sebagai mesin pencari vektor untuk mengambil informasi dari vectorstore.
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question}) #Memanggil metode atau fungsi langchain mengenai sistem percakapan dan pengambilan informasi
    st.session_state.chat_history = response['chat_history']            #Menyimpan hasil dari percakapan (chat history) ke dalam objek keadaan sesi
    # st.write(response)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:                                                  #Menggunakan kondisi untuk mengecek apakah indeks pesan saat ini adalah genap 
            st.write(user_template.replace(                             #Menampilkan pesan pengguna
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(                              #Menampilkan pesan balasan dari sistem atau bot yang diambil dari atribut content pada objek message.
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chatbot UDINUS",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.image('udinus.png',width=125)   
    st.header(":books: CHATBOT DOKUMEN AKADEMIK UDINUS :books:")
    user_question = st.text_input("Berikan pertanyaanmu :")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.image('udinus.png',width=125)   
        st.subheader("Dokumen Saya")
        pdf_docs = st.file_uploader(
            "Upload pdf lalu klik Process'", accept_multiple_files=True)
        try:
            if st.button("Process"):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success("Processing completed successfully!")

        except Exception as e:
            st.error(f"Processing Failed")

            
if __name__ == '__main__':
    main()
