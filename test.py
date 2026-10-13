from langchain_openai import ChatOpenAI

# Initialize the Qwen LLM with your custom URL
llm = ChatOpenAI(
    model="Qwen/Qwen3.8-27B-NVFP4",  # Must match the model name registered on your server
    api_key="API_k",  # Use "EMPTY" or a dummy string if your server doesn't require auth
    base_url="http://spark.mit.bme.hu:8888/v1",  # Your custom URL (usually requires /v1)
    temperature=0.7,
)

# Test invocation
response = llm.invoke("Hello! Introduce yourself briefly.")
print(response.content)
