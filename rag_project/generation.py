"""
generation.py
=============
LLM answer generation module — supports multiple CPU-friendly models.
"""

from transformers import pipeline, T5ForConditionalGeneration, AutoTokenizer
import torch

# Available LLMs — all CPU friendly
LLM_MODELS = {
    "distilgpt2": {
        "name": "distilgpt2",
        "type": "causal",
        "description": "Fast GPT-2 distilled model — 82M params"
    },
    "facebook/opt-125m": {
        "name": "facebook/opt-125m",
        "type": "causal",
        "description": "OPT 125M by Meta — GPT style open model"
    },
    "google/flan-t5-small": {
        "name": "google/flan-t5-small",
        "type": "seq2seq",
        "description": "Flan-T5 Small by Google — instruction tuned, best answers"
    }
}

# Cache loaded pipelines
_pipeline_cache = {}


def get_pipeline(model_name: str):
    """Load and cache an LLM pipeline."""
    if model_name in _pipeline_cache:
        return _pipeline_cache[model_name]

    info = LLM_MODELS.get(model_name, {})
    model_type = info.get("type", "causal")

    if model_type == "seq2seq":
        pipe = pipeline("text2text-generation", model=model_name)
    else:
        pipe = pipeline("text-generation", model=model_name)

    _pipeline_cache[model_name] = (pipe, model_type)
    return pipe, model_type


def generate_answer(question: str, context: str, model_name: str = "distilgpt2") -> str:
    """
    Generate an answer using the selected LLM.

    Args:
        question: User's question
        context: Retrieved context from vector DB
        model_name: LLM model name

    Returns:
        Generated answer string
    """
    pipe, model_type = get_pipeline(model_name)

    if model_type == "seq2seq":
        # Flan-T5 style prompt
        prompt = f"Answer the question based on the context below.\n\nContext: {context[:800]}\n\nQuestion: {question}\n\nAnswer:"
        result = pipe(prompt, max_new_tokens=150, do_sample=False)
        return result[0]["generated_text"].strip()
    else:
        # GPT style prompt
        prompt = f"""Answer ONLY using the given context. If not found, say: Not found in knowledge base.

Context:
{context[:600]}

Question: {question}

Answer:"""
        result = pipe(prompt, max_new_tokens=100, do_sample=False)
        output = result[0]["generated_text"]
        if "Answer:" in output:
            output = output.split("Answer:")[-1].strip()
        return output[:400]


def generate_all_models(question: str, context: str) -> dict:
    """
    Generate answers from all available LLMs for comparison.

    Returns:
        Dict with model_name as key and answer as value
    """
    results = {}
    for model_name in LLM_MODELS:
        try:
            answer = generate_answer(question, context, model_name)
            results[model_name] = answer
        except Exception as e:
            results[model_name] = f"Error: {str(e)}"
    return results
