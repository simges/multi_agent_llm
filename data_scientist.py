from typing_extensions import Annotated
from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent

from autogen_ext.models.ollama import OllamaChatCompletionClient
from dataclasses import dataclass
from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    message_handler,
    type_subscription,
)
from autogen_core.models import ModelFamily
from helpers.functions import clean_sql
import time
from autogen_agentchat.messages import TextMessage

data_scientist_config = {
    "model": "mistral-nemo:latest",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

g_final_sql = ""
g_db_name = ""

### 1.1 Create Tools
# a. SQL executor
async def execute_query() -> Annotated[str, "query results"]:
    import sqlite3

    global g_db_name, g_final_sql
    db_url = f"/home/simges/.cache/spider_data/test_database/{g_db_name}/{g_db_name}.sqlite"
    conn = sqlite3.connect(db_url)
    cursor = conn.cursor()
    try:
        cursor.execute(g_final_sql)
    except Exception as e:
        return "failure: " + str(e)
    return "passed"


@dataclass
class Message:
    content: str

# Constructor Output
class SQLOutput(BaseModel):
    sql: str

g_options = {
    "num_ctx": 16384,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "temperature": 0.0,
}

# CONSTRUCTOR CLIENT
data_scientist = OllamaChatCompletionClient(
    model=data_scientist_config["model"],
    host=data_scientist_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
    options=g_options,
    response_format=SQLOutput,
    max_tokens=300,
)


DATA_SCIENTIST_PROMPT = """
### Complete sqlite SQL query only and with no
explanation. 

### Sqlite database schema: 

{database_schema}

### {question}

"""


DATA_SCIENTIST_SYSTEM_PROMPT="You are a highly skilled AI assistant that translates \
natural language questions into correct and executable SQLite queries."
@type_subscription(topic_type="datascientist")
class DataScientist(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=data_scientist,
                                        system_message=DATA_SCIENTIST_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> str:
        print(f"{self.id.type} received message: {message.content}")
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        global g_final_sql
        g_final_sql = response.chat_message.content
        return g_final_sql


async def generate_query(question: str, schema: str, db_name: str) -> str:
    global g_db_name
    g_db_name = db_name
    runtime = SingleThreadedAgentRuntime()
    await DataScientist.register(runtime, "data_scientist", lambda: DataScientist("datascientist"))

    runtime.start()
    await runtime.send_message(
        Message(DATA_SCIENTIST_PROMPT.format(question=question, database_schema=schema)),
        recipient=AgentId(type="data_scientist", key="default"))
    await runtime.stop_when_idle()

    time.sleep(1)
    global g_final_sql
    g_final_sql = clean_sql(g_final_sql)
    print("Stripped SQL query: " + g_final_sql)

    result = await execute_query()
    print("Final execution result: " + result)
    return g_final_sql