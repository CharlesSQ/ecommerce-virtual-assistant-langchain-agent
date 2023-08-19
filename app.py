import time
import os
from config import OPENAI_API_KEY

from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import LLMSingleActionAgent, AgentExecutor, Tool
from langchain import LLMChain
from output_parser import ChatAgentOutputParser
from prompt_template import CustomPromptTemplate
from prompt import TEMPLATE_INSTRUCTIONS, SUFFIX
from agents.qa_agent import business_faq_agent

# Set API key for OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
print('API key set')

# Set OpenAI LLM and embeddings
llm_chat = ChatOpenAI(temperature=0.9, max_tokens=150,
                      model='gpt-3.5-turbo-0613', client='')

# Set conversation memory buffer
memory = ConversationBufferWindowMemory(
    memory_key="history", k=5, return_messages=True)

print('Memory buffer set')

# Set tools

tools = [
    Tool(
        name="get_business_faq",
        func=business_faq_agent.run,
        description="When you need to get information about a business. Input: send the user input.",
        # return_direct=True,
    )
]

# Set up prompt template
prompt = CustomPromptTemplate(
    prefix='',
    instructions=TEMPLATE_INSTRUCTIONS,
    sufix=SUFFIX,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"]
)

# Set up output parser
output_parser = ChatAgentOutputParser()

# Set up the agent
llm_chain = LLMChain(llm=llm_chat, prompt=prompt)
tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, max_iterations=3, memory=memory)


def main():
    user_prompt = input("Usuario: ")
    response = agent_executor.run(user_prompt)
    print('Assistant: ' + response)


if __name__ == '__main__':
    while True:
        main()
        time.sleep(1)
