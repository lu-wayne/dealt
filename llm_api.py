import openai
import requests


def call_llm(prompt, model="gpt-4o-mini", temperature=0.8, max_tokens=512, top_p=0.9):
    """
    Calls the specified large language model with given parameters.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"LLM API error: {e}")
        return None
