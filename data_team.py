import asyncio
import time
import os
from typing import Optional

from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    message_handler,
    type_subscription,
)
from autogen_agentchat.messages import TextMessage

from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from helpers.functions import execute_query, clean_sql
from helpers.ollama_clients import data_scientist, query_builder, query_builder2, refiner
from helpers.prompts import ANALYST_PROMPT, SCHEMA_LINKER_PROMPT, QUERY_BUILDER_PROMPT, REFINER_PROMPT

# Hardcoded paths for HuggingFace models and Tiktoken cache.
SENTENCE_TRANSFORMERS_HOME="/home/simges/.cache/huggingface/hub/sentence-transformers/{model_name}"
MINILM_EMBEDDING_MODEL_PATH=SENTENCE_TRANSFORMERS_HOME.format(model_name="/all-MiniLM-L6-v2")
os.environ["TIKTOKEN_CACHE_DIR"]="/home/simges/cl100k_base/"

Settings.embed_model = HuggingFaceEmbedding(model_name=MINILM_EMBEDDING_MODEL_PATH)


# --- Pydantic Models for Message and Output Formats ---
# These define the expected input and output structures for agent messages
# and LLM responses.
class Message(BaseModel):
    question: Optional[str] = None
    dbschema: Optional[str] = None
    content: Optional[str] = None


ANALYST_AGENT_SYSTEM_PROMPT="You are an assistant that extracts the user's goals \
    and target data from a natural language question about a database."
@type_subscription(topic_type="analyst")
class AnalystAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=data_scientist,
                                        system_message=ANALYST_AGENT_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")
        
        global g_reference_data, g_goals
        content=ANALYST_PROMPT.format(question=message.question,
                                database_schema=message.dbschema,
                                reference_data=g_reference_data)
        time.sleep(3)
        response = await self._delegate.on_messages([TextMessage(content=content, source="user")],
                                                     ctx.cancellation_token)

        print(f"{self.id.type} responded: {response.chat_message.content}")
        g_goals = "\n#USER GOALS\n" + response.chat_message.content
        await self.publish_message(message, topic_id=TopicId(type="schemalinker", source="analyst"))


SCHEMA_LINKER_SYSTEM_PROMPT="""You are a database-aware assistant you are tasked with \
    linking given user goals and question to the actual tables and columns in a database \
    schema. Explain your reasoning briefly. Do not return SQL."""
@type_subscription(topic_type="schemalinker")
class SchemaLinkerAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=data_scientist,
                                  system_message=SCHEMA_LINKER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        global g_goals, g_schemalink
        content=SCHEMA_LINKER_PROMPT.format(goals=g_goals, database_schema=message.dbschema)
        time.sleep(1)
        response = await self._delegate.on_messages([TextMessage(content=content, source="user")],
                                                    ctx.cancellation_token)

        print(f"{self.id.type} responded: {response.chat_message.content}")
        g_schemalink = "\n#SCHEMA LINKING\n" + response.chat_message.content
        await self.publish_message(message, topic_id=TopicId(type="qwenquerybuilder", source="schemalinker"))


BUILDER_SYSTEM_PROMPT = """You are a SQL builder. Build SQL query utilizing the information provided to you."""
@type_subscription(topic_type="qwenquerybuilder")
class QwenQueryBuilderAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._builder = AssistantAgent(name, model_client=query_builder, system_message=BUILDER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        global g_schemalink, g_goals, g_final_sql
        time.sleep(3)
        content=QUERY_BUILDER_PROMPT.format(question=message.question,
                                            schemalink=g_schemalink,
                                            goals=g_goals,
                                            database_schema=message.dbschema)
        response = await self._builder.on_messages([TextMessage(content=content, source="user")],
                                                    ctx.cancellation_token)

        print(f"{self.id.type} responded: {response.chat_message.content}")
        g_final_sql = clean_sql(response.chat_message.content)
        await self.publish_message(message, topic_id=TopicId(type="refiner", source="qwenquerybuilder"))


@type_subscription(topic_type="gemmaquerybuilder")
class GemmaQueryBuilderAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._builder = AssistantAgent(name, model_client=query_builder2, system_message=BUILDER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        time.sleep(3)
        global g_schemalink, g_goals, g_num_fix_attempts, g_final_sql
        g_num_fix_attempts = g_num_fix_attempts + 1


        content=QUERY_BUILDER_PROMPT.format(question=message.question,
                                            schemalink=g_schemalink,
                                            goals=g_goals,
                                            database_schema=message.dbschema)
        response = await self._builder.on_messages([TextMessage(content=content, source="user")],
                                                    ctx.cancellation_token)

        print(f"{self.id.type} responded: {response.chat_message.content}")
        g_final_sql = clean_sql(response.chat_message.content)
        await self.publish_message(message, topic_id=TopicId(type="refiner", source="gemmaquerybuilder"))


REFINER_SYSTEM_PROMPT = """You are tasked with correcting execution errors of SQL queries \
    based on error message and database schema. Check the query against database and fix \
    other incorrect column names. Explain explicitly how you fixed the error."""
@type_subscription(topic_type="refiner")
class RefinerAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=refiner, system_message=REFINER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        passed = False
        global g_num_fix_attempts, g_final_sql, g_db_name

        while g_num_fix_attempts < 5:
            result = await execute_query(g_final_sql, g_db_name)
            print(f"{self.id.type} execution result: {result}")

            if "passed" in result:
                passed = True
                break
            if g_num_fix_attempts == 2:
                break

            print(f"{self.id.type} received message: {message.question}")
            g_num_fix_attempts = g_num_fix_attempts + 1

            message.content = REFINER_PROMPT.format(schema=message.dbschema,
                                                    query=g_final_sql,
                                                    error=result,
                                                    goals=g_goals)
            time.sleep(3)
            response = await self._delegate.on_messages([TextMessage(content=message.content, source="user")],
                                                        ctx.cancellation_token)

            print(f"{self.id.type} responded: {response.chat_message.content}")
            g_final_sql = clean_sql(response.chat_message.content)
        

        if g_num_fix_attempts == 5:
            print(f"{self.id.type} execution result: {result} too much fix attempt.")
            return
        if passed == True:
            print(f"{self.id.type} execution result: {result} FIXED.")
            return
        
        # If neither successful nor max attempts reached (and still has attempts left),
        # it publishes to 'gemmaquerybuilder' for another try.
        # This logic means 'gemmaquerybuilder' is only hit after a *failed* refinement.
        await self.publish_message(message, topic_id=TopicId(type="gemmaquerybuilder", source="refiner"))


# --- Global Variables for Agent State ---
g_reference_data = ""
g_schemalink = ""
g_final_sql = ""
g_goals = ""
g_db_name = ""
g_num_fix_attempts = 0
async def generate_query(question: str, schema: str, db_name: str) -> str:
    """
    Main function to orchestrate the SQL generation and refinement process using AutoGen agents.

    Args:
        question (str): The natural language question from the user.
        schema (str): The database schema.
        db_name (str): The name of the database to connect to.

    Returns:
        str: The final (hopefully corrected) SQL query.
    """
    global g_db_name, g_schemalink, g_reference_data, g_goals, g_num_fix_attempts, g_final_sql

    # Reset global variables at the start of each call. This is crucial for avoiding state leakage
    # between calls if not using a class-based approach.
    g_db_name = db_name
    g_final_sql = ""
    g_schemalink = ""
    g_goals = ""
    g_reference_data = ""
    g_num_fix_attempts = 0

    print("g_db_name: " + g_db_name)
    print("schema : " + schema)


    # --- LlamaIndex Retrieval ---
    # Initializes SentenceSplitter and loads documents for retrieval.
    Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)

    # Hardcoded path for document loading.
    documents = SimpleDirectoryReader(f"./test_database/{g_db_name}/").load_data()
    vector_index = VectorStoreIndex.from_documents(documents)
    retriever = vector_index.as_retriever(similarity_top_k=3)

    # Retrieves relevant data based on the user question.
    response = retriever.retrieve(f"""{question}""")
    if len(response) != 0:
        g_reference_data = response[0].text # Uses only the top retrieved document.
        print(f"Retriever responded: {g_reference_data}")


    # --- AutoGen Agent Runtime Setup and Execution ---
    runtime = SingleThreadedAgentRuntime()

    # Registering agents. Note the lambda functions to create new instances.
    await AnalystAgent.register(runtime, "analyst_agent", lambda: AnalystAgent("analyst"))
    await SchemaLinkerAgent.register(runtime, "schema_linker", lambda: SchemaLinkerAgent("schemalinker"))

    # Two different query builder agents are registered.
    await GemmaQueryBuilderAgent.register(runtime, "gemma_sql_builder",
                                 lambda: GemmaQueryBuilderAgent("gemmaquerybuilder"))
    await QwenQueryBuilderAgent.register(runtime, "qwen_sql_builder",
                                 lambda: QwenQueryBuilderAgent("qwenquerybuilder"))
    await RefinerAgent.register(runtime, "refiner_agent", lambda: RefinerAgent("refiner"))

    runtime.start()    
    await runtime.publish_message(Message(dbschema=schema, question=question),
                                  topic_id=TopicId(type="analyst", source="default"))
    await runtime.stop_when_idle()


    print("Final SQL query: " + g_final_sql)
    result = await execute_query(g_final_sql, g_db_name)
    print("Final execution result: " + result)
    return g_final_sql