# # retriever.py
# def print_retrieved_chunks(retrieved_chunks: list, top_k: int = 15):
#     """
#     Prints the top_k retrieved PARENT chunks to the console for inspection.
#     """
#     if not retrieved_chunks:
#         print("\n--- No Chunks Retrieved ---\n")
#         return

#     num_to_print = min(top_k, len(retrieved_chunks))

#     # --- MODIFIED: Title reflects Parent Chunks ---
#     print(f"\n--- Top {num_to_print} Retrieved Parent Chunks (Post-Reranking) ---")

#     for idx, chunk in enumerate(retrieved_chunks[:num_to_print], 1):
#         text_content = chunk.get('text', '')
#         if not isinstance(text_content, str):
#             text_content = str(text_content)

#         # Snippet is longer as these are full parent docs
#         snippet = text_content.replace("\n", " ").strip() 

#         chapter = chunk.get('chapter', 'N/A')
#         # --- MODIFIED: Use shloka_no directly ---
#         shloka = chunk.get('shloka_no', 'N/A')
#         score = chunk.get('score', None) # This is now the RERANKER score

#         score_str = f"Rerank Score: {score:.4f}" if score is not None and score != 0.0 else "Score: N/A (or pre-rerank)"

#         print(f"\nChunk {idx} (Ch: {chapter}, Sh: {shloka}) - {score_str}:")
#         print(f"{snippet}...")
#         print(f"{'-'*60}")
#     print("-" * 30 + "\n")
def print_retrieved_chunks(retrieved_chunks: list, top_k: int = 15):
    """
    Prints the full parent chunks (untruncated) after reranking.
    """
    if not retrieved_chunks:
        print("\n--- No Chunks Retrieved ---\n")
        return

    num_to_print = min(top_k, len(retrieved_chunks))
    print(f"\n--- Top {num_to_print} Retrieved Parent Chunks (Post-Reranking) ---")

    for idx, chunk in enumerate(retrieved_chunks[:num_to_print], 1):
        text_content = chunk.get('text', '')
        if not isinstance(text_content, str):
            text_content = str(text_content)

        # Keep the entire text, including newlines
        snippet = text_content.strip()

        chapter = chunk.get('chapter', 'N/A')
        shloka = chunk.get('shloka_no', 'N/A')
        score = chunk.get('score', None)

        score_str = (
            f"Rerank Score: {score:.4f}" 
            if score is not None and score != 0.0 
            else "Score: N/A (or pre-rerank)"
        )

        print(f"\nChunk {idx} (Ch: {chapter}, Sh: {shloka}) - {score_str}:")
        print(snippet)  # <-- no truncation, no ellipsis
        print("-" * 60)

    print("-" * 30 + "\n")
