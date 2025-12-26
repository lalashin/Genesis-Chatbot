import langchain.agents
import inspect
import langchain

output = []
output.append(f"LangChain version: {langchain.__version__}")
try:
    src = inspect.getsource(langchain.agents.create_agent)
    output.append("Source of create_agent:")
    output.append(src)
except Exception as e:
    output.append(str(e))

with open("agent_source.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))
