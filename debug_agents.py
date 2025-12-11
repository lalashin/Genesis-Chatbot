import langchain.agents
with open("agent_dir.txt", "w") as f:
    f.write(str(dir(langchain.agents)))
