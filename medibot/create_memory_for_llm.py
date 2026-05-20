import json
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing. Add it to your .env file.")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_FAISS_PATH = "vectorstore/db_faiss"
MEMORY_FILE = Path("medibot_chat_history.json")
MAX_MEMORY_MESSAGES = 12


def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.4,
        max_new_tokens=512,
        top_p=0.95,
        repetition_penalty=1.05,
        return_full_text=False,
        huggingfacehub_api_token=HF_TOKEN,
    )


def load_vector_db():
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    return FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def load_chat_history():
    if not MEMORY_FILE.exists():
        return []

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            saved_messages = json.load(file)

        chat_history = []

        for item in saved_messages:
            role = item.get("role")
            content = item.get("content", "")

            if role == "human":
                chat_history.append(HumanMessage(content=content))
            elif role == "ai":
                chat_history.append(AIMessage(content=content))

        return chat_history[-MAX_MEMORY_MESSAGES:]

    except Exception:
        return []


def save_chat_history(chat_history):
    safe_history = chat_history[-MAX_MEMORY_MESSAGES:]

    saved_messages = []

    for message in safe_history:
        if isinstance(message, HumanMessage):
            saved_messages.append(
                {
                    "role": "human",
                    "content": message.content,
                }
            )
        elif isinstance(message, AIMessage):
            saved_messages.append(
                {
                    "role": "ai",
                    "content": message.content,
                }
            )

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(saved_messages, file, indent=2, ensure_ascii=False)


def clear_chat_history():
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()


def build_chat_chain():
    llm = load_llm()
    db = load_vector_db()

    retriever = db.as_retriever(
        search_kwargs={
            "k": 6,
        }
    )

    contextualize_question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are helping rewrite follow-up questions for medical knowledge retrieval.

Given the chat history and the latest user question, rewrite the latest question
as a complete standalone question.

Rules:
- Do not answer the question.
- Do not add new medical facts.
- Only rewrite the question if needed.
- If it is already standalone, return it unchanged.
""",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_question_prompt,
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are Medibot, a knowledgeable and helpful medical information assistant.

Use the retrieved context as your primary source to answer the user's question.
If the context is partially relevant, combine it with your general medical knowledge to give the most helpful answer possible.
Use the chat history only to understand references like "he", "she", "it", "that problem", or previous symptoms.
Only say "I don't know" if the question is completely outside the scope of medicine or health.

Rules:
- Answer the question fully using the context and your medical knowledge.
- Do not invent specific statistics or cite sources not in the context.
- Do not diagnose the user personally.
- Keep the answer clear, practical, and easy to understand.
- For personal medical decisions, recommend consulting a qualified healthcare professional.
- For emergency symptoms such as severe chest pain, trouble breathing, stroke signs, fainting, or severe bleeding, advise urgent medical care.

Retrieved context:
{context}
""",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(
        llm,
        answer_prompt,
    )

    return create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )


def print_sources(source_documents):
    if not source_documents:
        print("\nSources: No sources found.")
        return

    seen_sources = []

    for doc in source_documents:
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page")

        if page is not None:
            source_text = f"{source}, page {page}"
        else:
            source_text = str(source)

        if source_text not in seen_sources:
            seen_sources.append(source_text)

    print("\nSources:")

    for index, source in enumerate(seen_sources, start=1):
        print(f"{index}. {source}")


def print_intro(chat_history):
    print("\nMedibot is ready.")
    print("Type your question.")
    print("Type 'exit' to stop.")
    print("Type 'reset' to clear saved chat memory.")
    print("Type 'history' to see how many messages are remembered.")

    if chat_history:
        print(f"\nLoaded {len(chat_history)} saved chat messages.")
    else:
        print("\nNo saved chat memory found.")

    print()


def main():
    try:
        qa_chain = build_chat_chain()
    except Exception as error:
        print(f"\nFailed to start Medibot: {error}")
        return

    chat_history = load_chat_history()
    print_intro(chat_history)

    while True:
        user_query = input("You: ").strip()

        if not user_query:
            continue

        command = user_query.lower()

        if command in ["exit", "quit", "bye"]:
            save_chat_history(chat_history)
            print("\nBot: Goodbye!")
            break

        if command == "reset":
            chat_history.clear()
            clear_chat_history()
            print("\nBot: Chat memory cleared.\n")
            continue

        if command == "history":
            print(f"\nBot: I currently remember {len(chat_history)} chat messages.\n")
            continue

        try:
            response = qa_chain.invoke(
                {
                    "input": user_query,
                    "chat_history": chat_history,
                }
            )

            answer = response.get("answer", "No answer generated.")
            source_documents = response.get("context", [])

            print("\nBot:")
            print(answer)

            print_sources(source_documents)

            chat_history.append(HumanMessage(content=user_query))
            chat_history.append(AIMessage(content=answer))

            chat_history = chat_history[-MAX_MEMORY_MESSAGES:]
            save_chat_history(chat_history)

            print("\n" + "-" * 60 + "\n")

        except Exception as error:
            print(f"\nBot error: {error}\n")


if __name__ == "__main__":
    main()