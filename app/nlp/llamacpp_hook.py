from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.llms import LlamaCpp
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.output_parsers import PydanticOutputParser
from pydantic.v1 import BaseModel, Field, validator
from typing_extensions import TypedDict
from typing import List
import re
import json

from ..types import BucketGoalTemplate, MODEL_INFOS

# class BucketLists(BaseModel):
#     # bucketlists: List[TypedDict("bucketlists", {"1.": str, "2.": str, "3.": str, "4.": str, "5.": str})] = Field(description="a list of 5 small steps to achieve the ultimate goal.")
#     bucketlists: list[BucketGoalTemplate] = Field(description="큰 목표를 이루어나가기 위한 5가지 작은 단계를 제시해줘.")
#     # @validator("setup")
#     # def question_ends_with_question_mark(cls, field):
#     #     if field[-1] != "?":
#     #         raise ValueError("Badly formed question!")
#     #     return field

def parse_model_response(answer) -> list[BucketGoalTemplate]:
    pattern = r'\d+\.\s*(.*?)(?=\n\d+\.|\n*$)'

    result = re.sub(r'\*\*', '', answer)
    # matches = re.findall(pattern, result)
    matches = result.split("\n")

    # print(f"\n\n=====Formatting====\n{matches}")

    parsed_answer = []

    for item in matches:
        item = re.sub(r'\d+\.\d+', '', item).replace("\"", "").strip()
        split_idx = item.find("-")

        if split_idx == -1:
            parsed_answer.append(BucketGoalTemplate(name=item[2:], desc=""))
            # raise Exception("Answer Have Some Problem")
            continue
        
        name, desc = item[2:split_idx], item[split_idx+1:]
        name = name.strip()
        desc = desc.strip()
        parsed_answer.append(BucketGoalTemplate(name=name, desc=desc))
    
    return parsed_answer

class LLMAgent:
    def __init__(self, model_type, verbose=False):
        # base_template = """너는 버킷리스트를 이루는 것을 도와주는 역할을 맡은 멘토야. 큰 목표가 주어지면 너는 이 목표를 이루어나갈 수 있도록 큰 목표를 작은 목표들로 나눈 5단계를 제시해줘. 나의 목표는 "{question}"이고, 큰 목표를 이루어나가기 위한 5가지 작은 단계를 각 항목에 대한 간단한 설명과 함께 todo list를 작성하는 것과 유사하게 작성해줘."""
        # Using this reference when writing the steps. {search_result}

        base_template = """
            You are a mentor who helps me to achieve my bucket list. When you are given a big goal, please provide 5 steps that divide the big goal into small goals so that you can achieve it. Please write {num_items} small steps to achieve the big goal similar to writing a todo list with a brief description of each item in line by line. Small steps will be satisfied these conditions. First, it has to be achievable. And it has to be not hard to approach. 
            The output should be formatted as a structured list format like this, "1. 주제 - 간략한 설명". Essentially, You should split the title and brief description with "-".
            My goal is "{question}" Please write {num_items} steps that satisfy my requests and all interactions with me must be in Korean, Don't use english please. Please don't write start description but include list item description with splitter. Please don't list up the additional information of list object. 
        """

        # parser = PydanticOutputParser(pydantic_object=BucketLists)        
        # format_instructions = parser.get_format_instructions()

        # prompt = PromptTemplate.from_template(base_template)
        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(base_template),
            ],
            input_variables=["question", "search_result", "num_items"],
            # partial_variables={"format_instructions": format_instructions}
        )

        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Make sure the model path is correct for your system!
        llm = LlamaCpp(
            model_path=MODEL_INFOS[model_type],
            temperature=0,
            max_tokens=2048,
            top_p=1,
            f16_kv=True,
            n_gpu_layers=-1,
            callback_manager=callback_manager,
            verbose=verbose,  # Verbose is required to pass to the callback manager
        )

        # chain = prompt | llm | parser
        self.chain = prompt | llm

        self.search = DuckDuckGoSearchRun()

    def get_search_result(self, prompt):
        search_result = self.search.run(f"{prompt}를 이루기 위해서 필요한 단계")
        print(search_result)
        return search_result
    
    def get_answer(self, prompt, num_items=5, search=False, streaming=False):
        if search:
            search_result = self.get_search_result(prompt)
        else: search_result = ""

        processed = 0
        result = ""

        for s in self.chain.stream({"question": prompt, "num_items": num_items, "seasrch_result": search_result}):
            result += s

            if processed >= result.split("\n").__len__() - 1:
                continue
            
            parsed_resp = parse_model_response(result)
            if streaming:
                yield json.dumps([item.dict() for item in parsed_resp[:processed+1]]) + "\n"
            processed += 1

            if num_items == processed:
                break
        
        parsed_resp = parse_model_response(result)
        yield parsed_resp[:num_items]
            
# for s in chain.stream({"question": "수능 만점 받기"}):
    # print(s)

if __name__ == "__main__":
    agent = LLMAgent("mistral", True)
    agent.get_answer("수능 만점 받기")