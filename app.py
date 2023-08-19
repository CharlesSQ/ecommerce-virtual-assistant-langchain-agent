import time
import os
from config import OPENAI_API_KEY

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain import LLMMathChain
from langchain.agents import initialize_agent, Tool
from langchain.tools import MoveFileTool, format_tool_to_openai_function

# Set API key for OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    )
]
functions = [format_tool_to_openai_function(t) for t in tools]


def main():
    user_prompt = input("Usuario: ")
    response = llm.predict_messages(
        [HumanMessage(content=user_prompt)], functions=functions
    )
    print('Assistant: ', response)


if __name__ == '__main__':
    while True:
        main()
        time.sleep(1)
