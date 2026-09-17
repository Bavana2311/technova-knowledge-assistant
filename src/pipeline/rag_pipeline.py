from src.reranking.reranker import CrossEncoderReranker
from src.generation.llm_generator import GeminiGenerator
from src.memory.conversation_memory import (
    get_recent_memory,
    get_recent_summary,
    update_recent_memory
)


class RAGPipeline:

    def __init__(
        self,
        hybrid_retriever,
        reranker,
        generator,
        session_id="default"
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        self.session_id = session_id

    def query(self, question, top_k=5):

        # 1. Get conversation memory
        recent_memory = get_recent_memory(
            self.session_id
        )

        summary = get_recent_summary(
            self.session_id
        )

        # 2. Create retrieval query
        if recent_memory != "No recent messages found.":

            lines = recent_memory.split("\n")

            previous_questions = [
                line.replace("User: ", "").strip()
                for line in lines
                if line.startswith("User:")
            ]

            if previous_questions:

                previous_question = previous_questions[-1]

                retrieval_query = (
                    previous_question
                    + " "
                    + question
                )

            else:
                retrieval_query = question

        else:
            retrieval_query = question

        # 3. Hybrid retrieval
        hybrid_results = self.hybrid_retriever.retrieve(
            retrieval_query,
            top_k=top_k
        )

        # 4. Extract documents
        documents = [
            doc for doc, score in hybrid_results
        ]

        # 5. Reranking
        reranked_results = self.reranker.rerank(
            retrieval_query,
            documents,
            top_k=top_k
        )

        # 6. Remove clearly irrelevant results
        reranked_results = [
            (doc, score)
            for doc, score in reranked_results
            if score > 0
        ]

        # 7. Prepare context
        context_parts = []

        for i, (doc, score) in enumerate(
            reranked_results,
            start=1
        ):
            context_parts.append(
                f"[Source {i}]\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        # 8. Prepare conversation history
        conversation_history = f"""
Recent conversation:
{recent_memory}

Conversation summary:
{summary}
"""

        # 9. Generate answer
        answer = self.generator.generate(
            query=question,
            context=context,
            conversation_history=conversation_history
        )

        # 10. Update memory
        update_recent_memory(
            self.session_id,
            question,
            answer
        )

        # 11. Prepare sources
        sources = []

        for doc, score in reranked_results:

            sources.append({
                "source": doc.metadata.get(
                    "source",
                    "Unknown"
                ),
                "page": doc.metadata.get(
                    "page"
                ),
                "score": round(
                    float(score),
                    4
                )
            })

        # 12. Return result
        return {
            "answer": answer,
            "sources": sources
        }