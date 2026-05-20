import os
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
DB_FAISS_PATH = "vectorstore/db_faiss"


def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.4,
        max_new_tokens=512,
        repetition_penalty=1.05,
        return_full_text=False,
        huggingfacehub_api_token=HF_TOKEN,
    )


def load_vector_db():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def build_chat_chain():
    llm = load_llm()
    db = load_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 6})

    contextualize_question_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Given the chat history and the latest user question, rewrite the latest question
as a standalone question.

Only rewrite the question. Do not answer it.

If the latest question is already standalone, return it unchanged.
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
You are a knowledgeable and helpful medical assistant.

Use the retrieved context as your primary source to answer the user's question.
If the context is partially relevant, combine it with your general medical knowledge to give the most helpful answer possible.
You may also use the chat history to understand what the user is referring to.
Only say "I don't know" if the question is completely outside the scope of medicine or health.

Rules:
- Answer the question fully using the context and your medical knowledge.
- Do not invent specific statistics or cite sources not in the context.
- Do not diagnose the user personally.
- Keep the answer clear, practical, and easy to understand.
- For personal medical decisions, tell the user to consult a qualified healthcare professional.

Context:
{context}
""",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, answer_prompt)

    return create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )


def print_sources(source_documents):
    if not source_documents:
        print("\n📄 SOURCES: No sources found.")
        return

    print("\n📄 SOURCES:")

    for idx, doc in enumerate(source_documents, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page")

        if page is not None:
            print(f"{idx}. {source}, page {page}")
        else:
            print(f"{idx}. {source}")


def main():
    qa_chain = build_chat_chain()
    chat_history = []

    print("\n✅ Medibot is ready.")
    print("Type your question.")
    print("Type 'exit' to stop.")
    print("Type 'reset' to clear current chat memory.\n")

    while True:
        user_query = input("You: ").strip()

        if not user_query:
            continue

        if user_query.lower() in ["exit", "quit", "bye"]:
            print("\nBot: Goodbye!")
            break

        if user_query.lower() == "reset":
            chat_history.clear()
            print("\nBot: Chat memory cleared.\n")
            continue

        response = qa_chain.invoke(
            {
                "input": user_query,
                "chat_history": chat_history,
            }
        )

        answer = response.get("answer", "No answer generated.")
        source_documents = response.get("context", [])

        print("\n🔹 RESULT:")
        print(answer)

        print_sources(source_documents)

        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=answer))

        # Keeps the current chat memory small and fast.
        # This remembers the last 10 messages.
        if len(chat_history) > 10:
            del chat_history[:-10]

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()