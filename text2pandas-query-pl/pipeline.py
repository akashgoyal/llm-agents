from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import PandasInstructionParser

from modules import instruction_str, pandas_prompt_str, response_synthesis_prompt_str

def define_pipeline(df, llm1, llm2):

    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(instruction_str=instruction_str, df_str=df.head(5))
    pandas_output_parser = PandasInstructionParser(df)
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    qp = QP(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": llm1,
            "pandas_output_parser": pandas_output_parser,
            "response_synthesis_prompt": response_synthesis_prompt,
            "llm2": llm2,
        },
        verbose=True,
    )
    qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
    qp.add_links(
        [
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link(
                "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
            ),
            Link(
                "pandas_output_parser",
                "response_synthesis_prompt",
                dest_key="pandas_output",
            ),
        ]
    )
    # add link from response synthesis prompt to llm2
    qp.add_link("response_synthesis_prompt", "llm2")

    return qp


def visualise_pipeline(qp):
    from pyvis.network import Network
    net = Network(notebook=True, cdn_resources="in_line", directed=True)
    net.from_nx(qp.dag)
    net.show("text2sql_dag.html")