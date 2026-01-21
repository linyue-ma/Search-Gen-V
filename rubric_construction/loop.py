from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from collections import defaultdict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from openai import OpenAI
from tqdm import tqdm
import os
import re
import json
import torch
import random
import hdbscan
import requests
import argparse
import itertools



@dataclass
class Segment:
    segid: str
    text: str

    @property
    def docid(self) -> str:
        return self.segid.split("#")[0]

@dataclass
class Nugget:
    ''' Attributable Nugget '''
    text: str
    docids: list[str]

@dataclass
class ScoredNugget(Nugget):
    importance: str



TEST_CASES = [
    "why do older people vote more",
    # "why doesn't america ban guns",
    # "how were the lives of germans affected by ww1",
    # "are humans responsible for animal extinction",
    # "why should intellectual property be protected",
    # "why are health insurance premiums rising",
    # "was the korean war necessary",
    # "how safe are vaccinated people",
    # "how do people discover new music",
    # "what makes nursing a science",
    # "why did oil crash in 2015",
]




class PromptTemplateManager:

    @staticmethod
    def _sys_prompt_template_for_seg2q():
        return """You are an expert in query rewriting, able to write useful new queries based on relevant information.


Task Description:
Given a question, a query, and a passage, you need to generate new queries by modifying the given query based on the information in the given passage.


Background:
This is not a general query rewriting task; rather, it is a step in the task of mining ground truth information for the given question within a web corpus. The given question usually comes from long-form QA datasets or research-style question datasets, which require multiple information points to answer. The given query was generated during the mining process, and the given passage is exactly what was retrieved using this query.


Core Principles:
1. The information referenced from the given passage is usually related entities and modifiers associated with the given query, which were not considered in the query itself.
2. The rewriting actions can only be selected from the given Executable Rewriting Operations, with a maximum of three operations combined per rewrite.
3. The rewritten query needs to be semantically expanded, making it more likely to recall passages that contain ground truth information for the question but have not yet been mined.
4. As long as the rewritten query can establish a connection with the question, it is allowed, even if the semantic link between them is not strong. For example, if the question is "Why do older people vote more", and the query is "Policies on benefits for retirees, such as healthcare", the latter may retrieve relevant policies that could serve as potential reasons for the former. Therefore, a connection exists, and this query is good.


Executable Rewriting Operations:
1. Synonym replacement
2. Hypernym replacement
3. Hyponym replacement
4. Entity name fuzzification
5. Entity name specification
6. Switching between interrogative forms such as what/how/why
7. Add or modify constraints on the query


Output Format (two parts):
1) Short reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. You can freely express your thought process, but follow the steps below:
    - Recall the information from the given passage that is useful for the rewrite.
    - Analyze which rewriting operations need to be applied.
    - Execute the rewrite.
PS: Do not generate <reasoning> or </reasoning> inside the <reasoning> ... </reasoning> tags to avoid parsing errors.

2) Generate the final rewritten queries: After the </reasoning> tag, provide the final rewritten queries. You need to follow the requirements below:
    - Output one plain-text new query per line, with no other content.
    - Generate up to {max_num_new_queries} rewritten queries.
    - If no rewritten queries can be generated, output [None] directly."""


    @staticmethod
    def _sys_prompt_template_for_seg2q_strict():
        return """You are an expert in query rewriting, able to write useful new queries based on relevant information.


Task Description:
Given a question, a query, and a passage, you need to generate new queries by modifying the given query based on the information in the given passage.


Background:
This is not a general query rewriting task; rather, it is a step in the task of mining ground truth information for the given question within a web corpus. The given question usually comes from long-form QA datasets or research-style question datasets, which require multiple information points to answer. The given query was generated during the mining process, and the given passage is exactly what was retrieved using this query.


Core Principles:
1. The information referenced from the given passage is usually related entities and modifiers associated with the given query, which were not considered in the query itself.
2. The rewriting actions can only be selected from the given Executable Rewriting Operations, with a maximum of three operations combined per rewrite.
3. The rewritten query needs to be semantically expanded, making it more likely to recall passages that contain ground truth information for the question but have not yet been mined.
4. The rewritten query must remain strictly within the domain relevant to the given question, and must not introduce any unrelated queries.

Executable Rewriting Operations:
1. Synonym replacement
2. Hypernym replacement
3. Hyponym replacement
4. Entity name fuzzification
5. Entity name specification
6. Switching between interrogative forms such as what/how/why
7. Add or modify constraints on the query (i.e., time, location, topic, condition, etc.)


Output Format (two parts):
1) Short reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. You can freely express your thought process, but follow the steps below:
    - Recall the information from the given passage that is useful for the rewrite.
    - If there is useful/relevant information in the passage, analyze which rewriting operations need to be applied.
    - Execute the rewrite.
PS: Do not generate <reasoning> or </reasoning> inside the <reasoning> ... </reasoning> tags to avoid parsing errors.

2) Generate the final rewritten queries: After the </reasoning> tag, provide the final rewritten queries. You need to follow the requirements below:
    - Output one plain-text new query per line, with no other content.
    - Generate at most {max_num_new_queries} rewritten queries. 
    - If no rewritten queries can be generated, output [None] directly.
    - It is better to provide fewer or even zero queries than to include irrelevant or low-quality ones."""


    @staticmethod
    def _user_prompt_template_for_seg2q():
        return """Question: {question}

Query to be rewritten: {query}

Passage: {segment}"""


    @staticmethod
    def _sys_prompt_template_for_duplicate_judge():
        return """You are an expert in search query judgment, capable of identifying similar queries.


Task Description:
Given a rewritten query and a batch of existing queries, you need to determine whether the rewritten query is similar to any of the existing queries.


Background:
This task is part of an information mining process through query rewriting. The goal is to determine whether a newly rewritten query is similar to an existing query, in order to avoid redundant retrieval. The strategies for query rewriting include synonym replacement, hypernym/hyponym replacement, entity name fuzzification or specification, interrogative form transformation, modification or addition of constraints, and so on.


Core Principles:
1. The definition of "similar" is that a query shares the same entity names and constraints as an existing query.
2. Do not judge by deep semantics. Consider queries similar only if they look similar on the surface. For instance, 'older people' and 'elderly individuals' should be treated as different. Keeping such similar queries helps expand the semantic representation range of the retriever and thus avoid missing information.


Output Format (two parts):
1) Short reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. You can freely express your thought process to compare the newly rewritten query with each existing query whether they are similar. Do not generate <reasoning> or </reasoning> inside the <reasoning> ... </reasoning> tags to avoid parsing errors.

2) Generate the final decision: After the </reasoning> tag, provide the final decision. You need to follow the requirements below:
    - If the rewritten query is similar to any query in the existing queries, return True; otherwise, return False.
    - Do not generate any other content."""


    @staticmethod
    def _sys_prompt_template_for_duplicate_judge_strict():
        return """You are an expert in search query judgment, capable of identifying similar queries.


Task Description:
Given a rewritten query and a batch of existing queries, you need to determine whether the rewritten query is similar to any of the existing queries.


Background:
This task is part of an information mining process through query rewriting. The goal is to determine whether a newly rewritten query is similar to an existing query, in order to avoid redundant retrieval. The strategies for query rewriting include synonym replacement, hypernym/hyponym replacement, entity name fuzzification or specification, interrogative form transformation, modification or addition of constraints, and so on.


Core Principles:
1. The definition of "similar" is that the rewritten query shares the same entity names and constraints as an existing query.
2. Do not judge by deep semantics. Consider queries similar only if they look similar on the surface. For instance, "older people" and "elderly individuals" should be treated as different. Keeping such similar queries helps expand the semantic representation range of the retriever and thus avoid missing information.
2. If a query differs superficially from an existing query in terms of entity names or constraints but is semantically equivalent, it should also be considered similar. Such as "older people" and "elderly individuals".


Output Format (two parts):
1) Short reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. You can freely express your thought process to compare the newly rewritten query with each existing query whether they are similar. Do not generate <reasoning> or </reasoning> inside the <reasoning> ... </reasoning> tags to avoid parsing errors.

2) Generate the final decision: After the </reasoning> tag, provide the final decision. You need to follow the requirements below:
    - If the rewritten query is similar to any query in the existing queries, return True; otherwise, return False.
    - Do not generate any other content."""
    

    @staticmethod
    def _user_prompt_template_for_duplicate_judge():
        return """The rewritten query: {rewritten_query}

A batch of existing queries:
{existing_queries}"""
    

    @staticmethod
    def _sys_prompt_template_for_time_judge():
        return """You are a professional LLM Judge.


Task Description:
Given a query and a passage retrieved based on that query, you are asked to determine whether the passage satisfies the time constraint specified in the query.


Background:
This task is part of an information mining process through query rewriting. Since the topics being explored may differ from the creation time of the corpus, there is a risk of retrieving information that is temporally inconsistent with the query. The purpose of this task is to prevent the exposure of such information.


Output format (two parts):
1) Short reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. You can freely express your thought process, but follow the steps below:
    - Clarify the time constraint specified in the query. Options include:
        - No clear limit (e.g., "What is the policy on benefits for retirees?"). This means the query is not time-sensitive and accepts passages from any time.
        - In a specific year.
        - After a specific year.
        - Before a specific year.
        - Between two specific years.
    - Determine whether the passage contains information that does not comply with the time constraint specified in the query.
PS: Do not generate <reasoning> or </reasoning> inside the <reasoning> ... </reasoning> tags to avoid parsing errors.

2) Generate the final decision: After the </reasoning> tag, provide the final decision. You need to follow the requirements below:
    - If the passage meets the time constraint specified in the query, output True; otherwise, output False.
    - Do not generate any other content."""


    @staticmethod
    def _sys_prompt_template_for_time_judge_strict():
        return """You are a professional LLM Judge.


Task Description:
Given a query and a passage retrieved based on that query, you are asked to determine whether the passage satisfies the time constraint specified in the query.


Background:
This task is part of an information mining process through query rewriting. Since the topics being explored may differ from the creation time of the corpus, there is a risk of retrieving information that is temporally inconsistent with the query. The purpose of this task is to prevent the exposure of such information.


Output format (two parts):
1) Short reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. You can freely express your thought process, but follow the steps below:
    - Check whether the query contains any temporal features. If no temporal features are present, or if the query accepts information across a broad time range, then any passage can be considered to satisfy the time constraint. End reasoning.
    - If the query contains temporal features, determine the time scope of the query. Options include:
        - A specific point in time (e.g., a particular year or century).
        - A time range (which can be between two points in time, before a certain time, or after a certain time).
    - Then, based on the intent of the query, determine the type of time constraint that the passage needs to satisfy. Options include:
        - Strictly Constrained: The passage information must be strictly within the time range specified by the query. For example, the query “floods in Asia in 2015” requires the passage to contain information strictly from 2015.
        - Forward Time Extension: The passage may include information earlier than the time range specified by the query, emphasizing causes or background related to the query. For example, the query “what were the political causes of the 2015 oil crisis” accepts information from before 2015.
        - Backward Time Extension: The passage may include information later than the time range specified by the query, emphasizing effects or consequences of the query. For example, the query “impact of the 2008 financial crisis on the automotive industry” accepts information from after 2008.
    - Based on the determined type of time constraint, analyze whether the passage satisfies the corresponding requirement. End reasoning.
PS: Do not generate <reasoning> or </reasoning> inside the <reasoning> ... </reasoning> tags to avoid parsing errors.

2) Generate the final decision: After the </reasoning> tag, provide the final decision. You need to follow the requirements below:
    - If the passage meets the time constraint specified in the query, output True; otherwise, output False.
    - Do not generate any other content."""


    @staticmethod
    def _user_prompt_template_for_time_judge():
        return """Query: {query}

Passage: {segment}"""


    @staticmethod
    def _sys_prompt_template_for_creating_nuggets():
        return """You are NuggetCreator, an intelligent assistant that can generate atomic nuggets of information from a passage.


Task:
Given a question and a possibly relevant or useful passage, you need to generate atomic nuggets of information from the passage, so that the nuggets can be the gold information required to answer the question.


Core Principles:
1. Each generated nugget should be a complete and unique statement of a fact from the passage (a sentence of about 10-20 words).
2. A nugget should include a clear subject, verb, object, and if necessary, include constraint information such as time, location, topic, etc.
3. A nugget should avoid using pronouns such as "it".
4. A nugget is not simply a salient statement within the context, but also one that helps answer the question.


Output Format (two parts):
1) Short reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. You can freely express your thought process, but follow the steps below:
    - Identify key factual statements in the passage.
    - If there are complete statements, determine whether each factual statement is valuable in answering the given question by organizing the answer from multiple perspectives, and based on that, decide whether to consider it as a nugget.
PS: Do not generate <reasoning> or </reasoning> inside the <reasoning> ... </reasoning> tags to avoid parsing errors.
    
2) Generate the nuggets: After the </reasoning> tag, provide the nuggets. You need to follow the requirements below:
    - Output one plain-text nugget per line, with no other content.
    - Make sure you generate at most {creator_max_nuggets} nuggets (can be less or empty).
    - If no complete statement that is valuable to the question can be found in the passage, do not generate any low-quality nuggets, and just return [None] directly.
    - Do not explain and make sure there is no redundant information."""
    

    @staticmethod
    def _user_prompt_template_for_creating_nuggets():
        return """Question to be answered: {question}

Passage: {segment}"""


    @staticmethod
    def _sys_prompt_template_for_grouping_nuggets():
        return """You are NuggetGrouper, an intelligent assistant that can group similar and related nuggets.


Task:
Given a question and a list of nuggets, you need to group together the nuggets that are topically similar.


Background:
A nugget refers to a semantically complete factual statement (a sentence of about 10-20 words) that helps answer the given question. A nugget should include a clear subject, verb, object, and if necessary, include constraint information such as time, location, topic, etc. Since there may be multiple sources containing similar information, the nuggets may be similar or even duplicated. Therefore, similar nuggets need to be grouped together so that they can be merged in subsequent steps.


Core Principles:
1. "Topically similar" means that two or more nuggets share similar or related themes, entities, or subjects.
2. Sometimes, similarity should be judged at the semantic level: even if the surface wording differs, nuggets with the same underlying meaning are considered similar. For example, "older people" and "elderly individuals."


Output Format (two parts):
1) Short reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. You can freely express your thought process, but follow the steps below:
    - Starting from one nugget, identify other nuggets that are topically similar to it.
    - Record the ID numbers of nuggets judged to be similar, to keep track of which nuggets have been grouped together.
    - Then, repeat the process starting from other nuggets that have not yet been grouped, until all nuggets are assigned to a group.
    - If a nugget has no other nuggets similar to it, it forms its own group.
PS: Do not generate <reasoning> or </reasoning> inside the <reasoning> ... </reasoning> tags to avoid parsing errors.

2) Generate the final lists of grouped nuggets: After the </reasoning> tag, provide the ID number lists of grouped nuggets. You need to follow the requirements below:
    - Output one group per line as a list of ID numbers in Python list format, for example: [1, 2, 3]
    - Even if a group contains only one nugget (i.e., the nugget has no other topically similar nuggets), output it in the same format, for example: [1]
    - Do not explain and make sure there is no redundant information."""


    @staticmethod
    def _user_prompt_template_for_grouping_nuggets():
        return """Question: {question}

List of nuggets:
{nuggets}"""


    @staticmethod
    def _sys_prompt_template_for_merging_nuggets():
        return """You are NuggetMerger, an intelligent assistant that can combine similar nuggets.


Task:
Given a question and a list of nuggets (each nugget corresponds to a ID number), you need to combine similar nuggets if necessary.


Background:
A nugget refers to a semantically complete factual statement (a sentence of about 10-20 words) that helps answer the given question. A nugget should include a clear subject, verb, object, and if necessary, include constraint information such as time, location, topic, etc. Since there may be multiple sources containing similar information, the nuggets may be similar or even duplicated.


Core Principles:
1. "Similar" means that two or more nuggets point to the same factual statement at the semantic level.
2. Merge similar nuggets into a single nugget, making sure it is the best and most complete description of the factual statement.
3. When merging, ensure that the merged nugget is not too long (more than 20 words) and does not lose any useful information.


Output Format (two parts):
1) Short reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. You can freely express your thought process, but follow the steps below:
    - Identify whether there are similar nuggets.
    - If there are similar nuggets and merging them would not make the merged nugget too long (more than 20 words), group the nuggets that need to be merged together, and record the ID numbers of the nuggets in each group.
    - For each group, merge and rewrite the nuggets into a single nugget.
PS: Do not generate <reasoning> or </reasoning> inside the <reasoning> ... </reasoning> tags to avoid parsing errors.

2) Generate the final merged nuggets: After the </reasoning> tag, provide the final merged nuggets. You need to follow the requirements below:
    - Output one plain-text merged nugget per line, following the indication of the ID numbers of the original nuggets that are merged into it. Example: nugget_text [1, 2, ...]
    - When nuggets are merged, the nuggets that are not merged should still follow the format of indicating their original ID numbers.
    - If there are no similar nuggets in the list, which means that no merging is needed, simply return: [NO NEED].
    - Do not explain and make sure there is no redundant information."""
    

    @staticmethod
    def _user_prompt_template_for_merging_nuggets():
        return """Question: {question}

List of nuggets:
{nuggets}"""
    

    @staticmethod
    def _sys_prompt_template_for_scoring_nuggets():
        return """You are NuggetScorer, an intelligent assistant that can label a list of nuggets based on their importance to a question.


Task:
Given a question and a list of nuggets, you need to label each of the {num_nuggets} nuggets either a "vital" or "okay" based on the following core principles.


Background:
A nugget refers to a semantically complete factual statement (a sentence of about 10 words) that can be the gold information required to answer the given question.


Core Principles:
1. A "vital" nugget represents a factual statement that must be present in a "good" answer, whether it pertains to the overall question or a specific aspect.
2. An "okay" nugget contributes worthwhile information about the question but is not essential; in other words, it is "good to have" but not mandatory.


Output Format (two parts):
1) Short reasoning: Place ALL your reasoning analysis inside <reasoning> ... </reasoning> tags. You can freely express your thought process about the reasons why each nugget is "vital" or "okay". Do not generate <reasoning> or </reasoning> inside the <reasoning> ... </reasoning> tags to avoid parsing errors.

2) Generate the final labels: After the </reasoning> tag, provide the final labels. You need to follow the requirements below:
    - Output the label of each nugget on a separate line.
    - The label must be either vital or okay, in plain text only, with no other content.
    - Do not explain and make sure there is no redundant information."""


    @staticmethod
    def _user_prompt_template_for_scoring_nuggets():
        return """Question: {question}

List of nuggets:
{nuggets}"""


    @staticmethod
    def get_messages_for_seg2q(
        question: str,
        query: str,
        segment: str,
        max_num_new_queries: int=3
    ):
        return [
            {
                "role": "system",
                "content": PromptTemplateManager._sys_prompt_template_for_seg2q_strict().format(max_num_new_queries=max_num_new_queries)
            },
            {
                "role": "user",
                "content": PromptTemplateManager._user_prompt_template_for_seg2q().format(question=question, query=query, segment=segment)
            }
        ]
    
    
    @staticmethod
    def parse_seg2q_response(response: str) -> list[str]:
        response = re.sub(r"<reasoning>.*?</reasoning>", "", response, flags=re.DOTALL).strip()
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        if len(lines) == 1 and lines[0].strip().upper() == "[NONE]":
            return None
        return lines


    @staticmethod
    def get_messages_for_duplicate_judge(
        rewritten_query: str,
        existing_queries: list[str],
    ):
        return [
            {
                "role": "system",
                "content": PromptTemplateManager._sys_prompt_template_for_duplicate_judge_strict()
            },
            {
                "role": "user",
                "content": PromptTemplateManager._user_prompt_template_for_duplicate_judge().format(rewritten_query=rewritten_query, existing_queries="\n".join(existing_queries))
            }
        ]
    

    @staticmethod
    def parse_duplicate_judge_response(response: str) -> bool:
        response = re.sub(r"<reasoning>.*?</reasoning>", "", response, flags=re.DOTALL).strip()
        if "true" in response.lower():
            return True
        else:
            return False
    

    @staticmethod
    def get_messages_for_time_judge(
        query: str,
        segment: str,
    ):
        return [
            {
                "role": "system",
                "content": PromptTemplateManager._sys_prompt_template_for_time_judge_strict()
            },
            {
                "role": "user",
                "content": PromptTemplateManager._user_prompt_template_for_time_judge().format(query=query, segment=segment)
            }
        ]
    

    @staticmethod
    def parse_time_judge_response(response: str) -> bool:
        response = re.sub(r"<reasoning>.*?</reasoning>", "", response, flags=re.DOTALL).strip()
        if "true" in response.lower():
            return True
        else:
            return False
    

    @staticmethod
    def get_messages_for_creating_nuggets(
        question: str,
        segment: str,
        creator_max_nuggets: int=3,
    ):
        return [
            {
                "role": "system",
                "content": PromptTemplateManager._sys_prompt_template_for_creating_nuggets().format(creator_max_nuggets=creator_max_nuggets)
            },
            {
                "role": "user",
                "content": PromptTemplateManager._user_prompt_template_for_creating_nuggets().format(question=question, segment=segment)
            }
        ]
    

    @staticmethod
    def parse_creating_nuggets_response(response: str) -> list[str]:
        response = re.sub(r"<reasoning>.*?</reasoning>", "", response, flags=re.DOTALL).strip()
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        if len(lines) == 1 and lines[0].strip().upper() == "[NONE]":
            return None
        return lines
    

    @staticmethod
    def get_messages_for_grouping_nuggets(
        question: str,
        nuggets: list[Nugget],
    ):
        return [
            {
                "role": "system",
                "content": PromptTemplateManager._sys_prompt_template_for_grouping_nuggets()
            },
            {
                "role": "user",
                "content": PromptTemplateManager._user_prompt_template_for_grouping_nuggets().format(question=question, nuggets="\n".join([f"[{idx}] {nugget.text}" for idx, nugget in enumerate(nuggets)]))
            }
        ]
    

    @staticmethod
    def parse_grouping_nuggets_response(response: str) -> list[list[int]]:
        response = re.sub(r"<reasoning>.*?</reasoning>", "", response, flags=re.DOTALL).strip()
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        results = []
        for line in lines:
            matches = re.findall(r'\[([^\]]*)\]', line)
            if not matches or len(matches) != 1:
                raise ValueError(f"LLM Error at parse_grouping_nuggets_response: {line}")
            
            result = []
            content = matches[0]
            elements = [x.strip() for x in content.split(",") if x.strip()]
            for elem in elements:
                try:
                    result.append(int(elem))
                except ValueError:
                    pass
            results.append(result)
        return results


    @staticmethod
    def get_messages_for_merging_nuggets(
        question: str,
        nuggets: list[Nugget],
    ):
        return [
            {
                "role": "system",
                "content": PromptTemplateManager._sys_prompt_template_for_merging_nuggets()
            },
            {
                "role": "user",
                "content": PromptTemplateManager._user_prompt_template_for_merging_nuggets().format(question=question, nuggets="\n".join([f"[{idx}] {nugget.text}" for idx, nugget in enumerate(nuggets)]))
            }
        ]
    

    @staticmethod
    def parse_merging_nuggets_response(response: str) -> list[tuple[str, list[int]]]:
        response = re.sub(r"<reasoning>.*?</reasoning>", "", response, flags=re.DOTALL).strip()
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        if len(lines) == 1 and lines[0].strip().upper() == "[NO NEED]":
            return None
        
        result = []
        pattern = re.compile(r"^(.*)\s*\[(.*?)\]$")
        for line in lines:
            match = pattern.match(line)
            if match:
                nugget_text = match.group(1).strip()
                id_str = match.group(2).strip()
                if id_str:
                    try:
                        ids = [int(x.strip()) for x in id_str.split(",") if x.strip()]
                    except Exception:
                        ids = []
                else:
                    ids = []
                result.append((nugget_text, ids))
        return result
    

    @staticmethod
    def get_messages_for_scoring_nuggets(
        question: str,
        nuggets: list[Nugget],
    ):
        return [
            {
                "role": "system",
                "content": PromptTemplateManager._sys_prompt_template_for_scoring_nuggets().format(num_nuggets=len(nuggets))
            },
            {
                "role": "user",
                "content": PromptTemplateManager._user_prompt_template_for_scoring_nuggets().format(question=question, nuggets="\n".join([nugget.text for nugget in nuggets]))
            }
        ]
    

    @staticmethod
    def parse_scoring_nuggets_response(response: str) -> list[str]:
        response = re.sub(r"<reasoning>.*?</reasoning>", "", response, flags=re.DOTALL).strip()
        return [line.strip() for line in response.split("\n") if line.strip()]




@dataclass
class BaseNode:
    """Base node type for the query-segment tree."""

    node_id: int
    depth: int
    parent_id: Optional[int]
    children_ids: List[int] = field(default_factory=list)

    def to_shallow_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "type": self.__class__.__name__,
        }

@dataclass
class QueryNode(BaseNode):
    """Node representing a query string in the information exploration tree."""
    
    query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = self.to_shallow_dict()
        data.update({"query": self.query})
        return data

@dataclass
class SegmentNode(BaseNode):
    """Node representing a retrieved segment in the information exploration tree."""

    segment: Segment = field(default_factory=lambda: Segment(segid="", text=""))
    nuggets: List[Nugget] = field(default_factory=list)
    source_query: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        data = self.to_shallow_dict()
        data.update(
            {
                "segment": {
                    "segid": self.segment.segid,
                    "text": self.segment.text,
                },
                "nuggets": [nugget.to_dict() for nugget in self.nuggets],
                "source_query": self.source_query,
            }
        )
        return data


class QueryPassageTree:

    def __init__(self):
        self._id_gen = itertools.count(start=1)
        self.nodes: Dict[int, BaseNode] = {}
        self.root_id: Optional[int] = None
    
    def _next_id(self):
        return next(self._id_gen)
    
    def add_query_node(self, query: str, depth: int, parent_id: Optional[int]) -> int:
        node_id = self._next_id()
        node = QueryNode(node_id=node_id, depth=depth, parent_id=parent_id, query=query)
        self.nodes[node_id] = node
        if parent_id is not None:
            self.nodes[parent_id].children_ids.append(node_id)
        else:
            self.root_id = node_id
        return node_id
    
    def add_segment_node(self, segment: Segment, depth: int, parent_id: int, source_query: str) -> int:
        node_id = self._next_id()
        node = SegmentNode(node_id=node_id, depth=depth, parent_id=parent_id, segment=segment, source_query=source_query)
        self.nodes[node_id] = node
        self.nodes[parent_id].children_ids.append(node_id)
        return node_id
    
    def iter_nodes(self) -> List[BaseNode]:
        return list(self.nodes.values())
    
    def iter_query_nodes(self) -> List[QueryNode]:
        return [n for n in self.nodes.values() if isinstance(n, QueryNode)]
    
    def iter_segment_nodes(self) -> List[SegmentNode]:
        return [n for n in self.nodes.values() if isinstance(n, SegmentNode)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_id": self.root_id,
            "nodes": {nid: (node.to_dict() if hasattr(node, "to_dict") else node.to_shallow_dict()) for nid, node in self.nodes.items()},
        }
    
    def to_json(self, ensure_ascii: bool = False, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)


class LLM:
    def __init__(self, model_name: str, model_api_url, model_api_key):
        self.model_name = model_name
        self.model_api_url = model_api_url
        self.model_api_key = model_api_key
        self.llm_client = self._initialize_client()

    def _initialize_client(self):
        return OpenAI(
            base_url=self.model_api_url,
            api_key=self.model_api_key
        )
    
    def call(self, messages: list[dict[str, str]], temperature: float = 0.7):
        max_trails = 50
        for _ in range(max_trails):
            try:
                completion = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=0.95,
                    max_completion_tokens=2048,
                    timeout=30
                )
                response = completion.choices[0].message.content
                return response
            except Exception as e:
                print(f"[LLM]LLM Error: {str(e)}")
        return None


class Embedder:
    def __init__(self,
                 model_name,
                 device: str="cpu",
                 max_seq_length: int=512):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.model.max_seq_length = max_seq_length
        self.model.eval()
    
    def encode(self, sentences, batch_size: int=16):
        with torch.no_grad():
            embeddings = self.model.encode(sentences,
                                           show_progress_bar=True,
                                           batch_size=batch_size,
                                           convert_to_tensor=False,
                                           normalize_embeddings=True)
        return embeddings


class Reranker:
    def __init__(self, model_path: str, max_seq_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self._model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        self._model.eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = max_seq_length

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
    
    def _format_instruction(self, query: str, text: str, instruction: str = None):
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction, query=query, doc=text)
        return output
    
    def _process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation='longest_first',
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self._model.device)
        return inputs
    
    @torch.no_grad()
    def _compute_logits(self, inputs, **kwargs):
        batch_scores = self._model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def calc_score(self, query: str, text: list[str], batch_size: int = 4):
        scores = []
        for i in tqdm(range(0, len(text), batch_size), desc="Reranker: Calculating scores"):
            batch_text = text[i: i + batch_size]
            batch_pairs = [self._format_instruction(query, t) for t in batch_text]
            batch_inputs = self._process_inputs(batch_pairs)
            batch_scores = self._compute_logits(batch_inputs)
            scores.extend(batch_scores)
        return scores


class TreeBuilder:

    def __init__(self, config):
        self.config = config
        self.retriever_url = config.retriever_url
        self.seg2q_model = LLM(config.seg2q_model, config.seg2q_model_api_url, config.seg2q_model_api_key)
        self.seg2q_judge_model = LLM(config.seg2q_judge_model, config.seg2q_judge_model_api_url, config.seg2q_judge_model_api_key)
        self.seg_judge_model = LLM(config.seg_judge_model, config.seg_judge_model_api_url, config.seg_judge_model_api_key)
        
        self.instruction = "Given a web search query, retrieve relevant passages that answer the query"
        self.template = "Instruct: {instruction}\nQuery: {query}"
    
    def _call_search(self, query: str, k: int) -> list[dict]:
        url = f"{self.retriever_url}/search"
        max_trails = 50
        for _ in range(max_trails):
            try:
                response = requests.post(url, json={"query": query, "k": k})
                return response.json()["hits"]
            except Exception as e:
                print(f"[TreeBuilder]Retriever Error: {str(e)}\n\n")
        print(f"[TreeBuilder]Retriever Error: Failed to search the query after {max_trails} trails.\n\n")
        return []
    
    def _call_fetch(self, segid: str) -> dict:
        url = f"{self.retriever_url}/fetch"
        max_trails = 50
        for _ in range(max_trails):
            try:
                response = requests.post(url, json={"segid": segid})
                return response.json()["segment"]
            except Exception as e:
                print(f"[TreeBuilder]Retriever Error: {str(e)}\n\n")
        print(f"[TreeBuilder]Retriever Error: Failed to fetch the segment after {max_trails} trails.\n\n")
        return None
    
    def _rewrite_query(self, question: str, query: str, segment: str, max_new_queries: int) -> list[str]:
        max_trails = 50
        messages = PromptTemplateManager.get_messages_for_seg2q(question, query, segment, max_new_queries)
        for _ in range(max_trails):
            try:
                response = self.seg2q_model.call(messages)
                queries = PromptTemplateManager.parse_seg2q_response(response)
                if queries is None:
                    return []
                return queries[:max_new_queries]
            except Exception as e:
                print(f"[TreeBuilder]LLM Error: {str(e)}\n\n")
        print(f"[TreeBuilder]LLM Error: Failed to rewrite the query after {max_trails} trails.\n\n")
        return []
    
    def _is_duplicate_query(self,
                           new_query: str,
                           existing_queries: list[str],
                           batch_size: int = 5) -> bool:
        if not new_query.strip():
            # new_query is empty
            return True

        start = 0
        while start < len(existing_queries):
            end = min(start + batch_size, len(existing_queries))
            batch_queries = existing_queries[start: end]
            messages = PromptTemplateManager.get_messages_for_duplicate_judge(new_query, batch_queries)
            max_trails = 50
            while max_trails > 0:
                try:
                    response = self.seg2q_judge_model.call(messages, temperature=0.2)
                    result = PromptTemplateManager.parse_duplicate_judge_response(response)
                    if result:
                        return True
                    else:
                        break
                except Exception as e:
                    print(f"[TreeBuilder]LLM Error: {str(e)}\n\n")
                    max_trails -= 1
                    if max_trails == 0:
                        print(f"[TreeBuilder]LLM Error: Failed to judge the query after {max_trails} trails.\n\n")
                        return True
            
            start += batch_size
        
        return False
    
    def _retrieve_segments(self, query: str, k: int, threshold: float = None, hard_k: int = 30) -> list[Segment]:
        # hard_k > k
        hits = self._call_search(query, hard_k)
        res = []
        if threshold is None:
            for i in range(k):
                segid = hits[i]["docid"]
                seg = self._call_fetch(segid)
                res.append({
                    "segid": segid,
                    "segment": seg["segment"],
                    "retrieval_score": hits[i]["score"],
                })
        else:
            for i in range(hard_k):
                if hits[i]["score"] > threshold:
                    segid = hits[i]["docid"]
                    seg = self._call_fetch(segid)
                    res.append({
                        "segid": segid,
                        "segment": seg["segment"],
                        "retrieval_score": hits[i]["score"],
                    })
                else:
                    break
        
        return [Segment(segid=seg["segid"], text=seg["segment"]) for seg in res]
    
    def _is_temporal_consistent(self, query: str, segment: str):
        max_trails = 50
        messages = PromptTemplateManager.get_messages_for_time_judge(query, segment)
        for _ in range(max_trails):
            try:
                response = self.seg_judge_model.call(messages, temperature=0.2)
                result = PromptTemplateManager.parse_time_judge_response(response)
                if result:
                    return True
                else:
                    return False
            except Exception as e:
                print(f"[TreeBuilder]LLM Error: {str(e)}\n\n")
        
        return False
    
    def build(self, question):
        tree = QueryPassageTree()

        existing_queries: list[str] = []
        seen_segment_ids: set[str] = set()

        root_id = tree.add_query_node(query=question, depth=0, parent_id=None)
        existing_queries.append(question)

        num_nodes_created = 1

        initial_segments = self._retrieve_segments(question, 
                                                   self.config.k_retrieved_passages,
                                                   self.config.threshold_score,
                                                   self.config.hard_k_retrieved_passages)
        
        for seg in initial_segments:
            if seg.segid in seen_segment_ids:
                print(f"[TreeBuilder]Skipping segment: {seg.segid} (already seen)\n\n")
                continue
            
            if not self._is_temporal_consistent(question, seg.text):
                print(f"[TreeBuilder]Skipping segment: {seg.segid} (temporal inconsistency)\nquery: {question}\nsegment: {seg.text}\n\n")
                continue

            seen_segment_ids.add(seg.segid)
            tree.add_segment_node(seg, depth=1, parent_id=root_id, source_query=question)
            num_nodes_created += 1
            if num_nodes_created >= self.config.max_total_nodes:
                return tree
        
        frontier: list[tuple[int, SegmentNode]] = []
        for child_id in tree.nodes[root_id].children_ids:
            child = tree.nodes[child_id]
            if isinstance(child, SegmentNode):
                frontier.append((child.depth, child))
        
        # 开始BFS扩展
        while frontier:
            print(f"*" * 60)
            print(f"[TreeBuilder]Unprocessed segment nodes: {len(frontier)}")
            print(f"[TreeBuilder]Total nodes: {len(tree.nodes)}")
            print(f"[TreeBuilder]Query nodes: {len(tree.iter_query_nodes())}")
            print(f"[TreeBuilder]Segment nodes: {len(tree.iter_segment_nodes())}")
            print(f"*" * 60)

            if num_nodes_created >= self.config.max_total_nodes:
                print(f"[TreeBuilder]Max total nodes reached: {num_nodes_created}\n\n")
                break

            current_depth, segment_node = frontier.pop(0)
            if current_depth >= self.config.max_depth:
                continue

            base_depth_for_queries = current_depth + 1

            derived_queries: list[str] = []
            try:
                derived_queries = self._rewrite_query(
                    question,
                    segment_node.source_query,
                    segment_node.segment.text,
                    self.config.max_queries_per_passage
                )
            except Exception as e:
                print(f"[TreeBuilder]Error: {str(e)}")
                derived_queries = []

            print(f"[TreeBuilder]Derived queries:\nsource_query: {segment_node.source_query}\nsegment: {segment_node.segment.text}\nderived_queries: {derived_queries}\n\n")
            
            unique_derived: list[str] = []
            for q in derived_queries:
                if self._is_duplicate_query(q, existing_queries):
                    print(f"[TreeBuilder]Skipping duplicate query: {q}\n\n")
                    continue
                else:
                    print(f"[TreeBuilder]Adding new query which is not duplicate: {q}\n\n")
                    unique_derived.append(q)

            for dq in unique_derived:
                if num_nodes_created >= self.config.max_total_nodes:
                    print(f"[TreeBuilder]Max total nodes reached: {num_nodes_created}\n\n")
                    break

                qnode_id = tree.add_query_node(query=dq, depth=base_depth_for_queries, parent_id=segment_node.node_id)
                existing_queries.append(dq)
                num_nodes_created += 1

                if base_depth_for_queries >= self.config.max_depth:
                    print(f"[TreeBuilder]Max depth reached: {base_depth_for_queries}\n\n")
                    continue

                new_segments = self._retrieve_segments(dq,
                                                       self.config.k_retrieved_passages,
                                                       self.config.threshold_score,
                                                       self.config.hard_k_retrieved_passages)
                any_unseen_added = False
                for new_seg in new_segments:
                    if new_seg.segid in seen_segment_ids:
                        print(f"[TreeBuilder]Skipping segment: {new_seg.segid} (already seen)\n\n")
                        continue
                    
                    if not self._is_temporal_consistent(dq, new_seg.text):
                        print(f"[TreeBuilder]Skipping segment: {new_seg.segid} (temporal inconsistency)\nquery: {dq}\nsegment: {new_seg.text}\n\n")
                        continue
                    
                    seen_segment_ids.add(new_seg.segid)
                    tree.add_segment_node(new_seg, depth=base_depth_for_queries + 1, parent_id=qnode_id, source_query=dq)
                    num_nodes_created += 1
                    any_unseen_added = True
                    
                    if num_nodes_created >= self.config.max_total_nodes:
                        print(f"[TreeBuilder]Max total nodes reached: {num_nodes_created}\n\n")
                        break
                
                if num_nodes_created >= self.config.max_total_nodes:
                    print(f"[TreeBuilder]Max total nodes reached: {num_nodes_created}\n\n")
                    break

                if any_unseen_added:
                    for child_id in tree.nodes[qnode_id].children_ids:
                        child = tree.nodes[child_id]
                        if isinstance(child, SegmentNode):
                            frontier.append((child.depth, child))
        
        return tree



class Nuggetizer:

    def __init__(self, config):
        self.config = config
        self.nuggetizer_model = LLM(config.nuggetizer_model, config.nuggetizer_model_api_url, config.nuggetizer_model_api_key)
        self.embedder = Embedder(config.embedder_model)
    
    def _create_nuggets(self, question: str, segment: str, creator_max_nuggets: int=3):
        max_trails = 50
        messages = PromptTemplateManager.get_messages_for_creating_nuggets(question, segment, creator_max_nuggets)
        for _ in range(max_trails):
            try:
                response = self.nuggetizer_model.call(messages, temperature=0.2)
                nuggets = PromptTemplateManager.parse_creating_nuggets_response(response)
                if nuggets is None:
                    return []
                return nuggets[:creator_max_nuggets]
            except Exception as e:
                print(f"[Nuggetizer]LLM Error at _create_nuggets: {str(e)}")
        print(f"[Nuggetizer]LLM Error at _create_nuggets: Failed to generate nuggets after {max_trails} trails.")
        return []
    
    def _merge_nuggets_by_embedding(self, question: str, nuggets: list[Nugget]) -> list[Nugget]:
        text_list = [nugget.text for nugget in nuggets]
        embeddings = self.embedder.encode(text_list)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")
        clusterer.fit(embeddings)
        labels = clusterer.labels_

        merged_nuggets: list[Nugget] = []
        for cluster_label in set(labels):
            for i, label in enumerate(labels):
                if label == cluster_label:
                    print(f"[Nuggetizer]Nugget {i} in cluster {cluster_label}: {nuggets[i].text}")
        
        for cluster_label in tqdm(set(labels), desc="Merging nuggets by embedding"):
            if cluster_label == -1:
                for i, label in enumerate(labels):
                    if label == cluster_label:
                        merged_nuggets.append(nuggets[i])
            else:
                cluster_nuggets = []
                for i, label in enumerate(labels):
                    if label == cluster_label:
                        cluster_nuggets.append(nuggets[i])
                
                if len(cluster_nuggets) < 2:
                    for item in cluster_nuggets:
                        merged_nuggets.append(item)
                else:
                    if len(cluster_nuggets) > 30:
                        batches = defaultdict(list)
                        for nugget in cluster_nuggets:
                            first_word = nugget.text.split()[0].lower()
                            batches[first_word].append(nugget)
                        unbatched_nuggets = []

                        for _, batch in batches.items():
                            if len(batch) == 1:
                                unbatched_nuggets.append(batch[0])
                            elif len(batch) > 30:
                                for i in range(0, len(batch), 30):
                                    merged_nuggets.extend(self._merge_nuggets(question, batch[i: i + 30]))
                            else:
                                merged_nuggets.extend(self._merge_nuggets(question, batch))
                        
                        if len(unbatched_nuggets) > 0:
                            if len(unbatched_nuggets) == 1:
                                merged_nuggets.append(unbatched_nuggets[0])
                            elif len(unbatched_nuggets) > 30:
                                for i in range(0, len(unbatched_nuggets), 30):
                                    merged_nuggets.extend(self._merge_nuggets(question, unbatched_nuggets[i: i + 30]))
                            else:
                                merged_nuggets.extend(self._merge_nuggets(question, unbatched_nuggets))
                    else:
                        merged_nuggets.extend(self._merge_nuggets(question, cluster_nuggets))
        
        return merged_nuggets
    
    def _group_nuggets(self, question: str, nuggets: list[Nugget]) -> list[list[int]]:
        max_trails = 50
        grouped_nuggets: list[list[Nugget]] = []
        messages = PromptTemplateManager.get_messages_for_grouping_nuggets(question, nuggets)
        for _ in range(max_trails):
            try:
                response = self.nuggetizer_model.call(messages, temperature=0.2)
                print(f"[Nuggetizer]Grouping response: {response}")
                grouped_results = PromptTemplateManager.parse_grouping_nuggets_response(response)
                print(f"[Nuggetizer]Grouped results: {grouped_results}")

                ids_seen = []
                for i, grouped_result in enumerate(grouped_results):
                    current_group = []
                    for idx in grouped_result:
                        if idx in ids_seen:
                            raise ValueError(f"Nugget {idx} already seen in group {i}")
                        ids_seen.append(idx)

                        current_group.append(nuggets[idx])
                        print(f"[Nuggetizer] Nugget in group {i}: {nuggets[idx].text}")
                    print("\n\n")
                    grouped_nuggets.append(current_group)

                flat_grouped_nuggets = [idx for grouped_result in grouped_results for idx in grouped_result]
                ids_not_grouped = [idx for idx in range(len(nuggets)) if idx not in flat_grouped_nuggets]
                for idx in ids_not_grouped:
                    grouped_nuggets.append([nuggets[idx]])
                    print(f"[Nuggetizer] Nugget not grouped: {nuggets[idx].text}")

                break
            except Exception as e:
                print(f"[Nuggetizer]LLM Error at _group_nuggets: {str(e)}")
        
        return grouped_nuggets if len(grouped_nuggets) > 0 else nuggets
    
    def _merge_nuggets(self, question: str, nuggets: list[Nugget]) -> list[Nugget]:
        max_trails = 50
        merged_nuggets: list[Nugget] = []
        messages = PromptTemplateManager.get_messages_for_merging_nuggets(question, nuggets)
        for _ in range(max_trails):
            try:
                response = self.nuggetizer_model.call(messages, temperature=0.2)
                print(f"[Nuggetizer]Merging response: {response}")
                merged_results = PromptTemplateManager.parse_merging_nuggets_response(response)
                if merged_results is not None and len(merged_results) > 0:
                    for nugget_text, ids in merged_results:
                        if not isinstance(ids, list) or not all(isinstance(idx, int) for idx in ids):
                            raise ValueError(f"[Nuggetizer]warning: ids is not a list of int: {ids}")
                        if not isinstance(nugget_text, str):
                            raise ValueError(f"[Nuggetizer]warning: nugget_text is not a string: {nugget_text}")
                        docids = []
                        for idx in ids:
                            if 0 <= idx < len(nuggets):
                                docids.extend(nuggets[idx].docids)
                            else:
                                raise ValueError(f"[Nuggetizer]warning: idx out of range: {idx}")
                        merged_nuggets.append(Nugget(text=nugget_text, docids=docids))
                else:
                    merged_nuggets = nuggets[:]
                break
            except Exception as e:
                print(f"[Nuggetizer]LLM Error at _merge_nuggets: {str(e)}")
        
        return merged_nuggets
    
    def _group_then_merge_nuggets(self, question: str, nuggets: list[Nugget]) -> list[Nugget]:
        if not nuggets or len(nuggets) == 0:
            return []
        
        grouped_nuggets = self._group_nuggets(question, nuggets)
        merged_nuggets = []
        for group in grouped_nuggets:
            if len(group) == 1:
                merged_nuggets.append(group[0])
            elif len(group) > 1:
                merged_nuggets.extend(self._merge_nuggets(question, group))

        return merged_nuggets
    
    def _merge_nuggets_divide_and_conquer(self, question: str, nuggets: list[Nugget], window_size: int = 10) -> list[Nugget]:
        
        if not nuggets or len(nuggets) == 0:
            return []
        if window_size is None or window_size <= 0:
            window_size = 10

        current_nuggets: list[Nugget] = list(nuggets)
        previous_count: int = len(current_nuggets)
        consecutive_no_reduction_rounds: int = 0
        max_rounds: int = 10
        rounds: int = 0

        while True:
            rounds += 1
            print(f"[Nuggetizer]Merging nuggets by divide-and-conquer: {rounds} rounds")
            if rounds > max_rounds:
                break
            
            shuffled_nuggets = current_nuggets[:]
            random.shuffle(shuffled_nuggets)

            next_round_nuggets: list[Nugget] = []
            start = 0
            while start < len(shuffled_nuggets):
                end = min(start + window_size, len(shuffled_nuggets))
                batch = shuffled_nuggets[start:end]
                try:
                    merged_batch = self._merge_nuggets(question, batch)
                except Exception:
                    merged_batch = []
                
                if not merged_batch:
                    next_round_nuggets.extend(batch)
                else:
                    next_round_nuggets.extend(merged_batch)
                start += window_size

            current_count = len(next_round_nuggets)
            if current_count >= previous_count:
                consecutive_no_reduction_rounds += 1
            else:
                consecutive_no_reduction_rounds = 0

            current_nuggets = next_round_nuggets
            previous_count = current_count

            if consecutive_no_reduction_rounds >= 2:
                break

            if current_count == 1:
                break

        return current_nuggets
    
    def _score_nuggets(self, question: str, nuggets: list[Nugget], window_size: int = 10) -> list[ScoredNugget]:
        start = 0
        scored_nuggets: list[ScoredNugget] = []
        while start < len(nuggets):
            end = min(start + window_size, len(nuggets))
            batch_nuggets = nuggets[start: end]
            messages = PromptTemplateManager.get_messages_for_scoring_nuggets(question, batch_nuggets)
            max_trails = 50
            for _ in range(max_trails):
                try:
                    response = self.nuggetizer_model.call(messages, temperature=0.2)
                    scored_results = PromptTemplateManager.parse_scoring_nuggets_response(response)
                    for nugget, label in zip(batch_nuggets, scored_results):
                        scored_nuggets.append(ScoredNugget(text=nugget.text, docids=nugget.docids, importance=label.lower()))
                    break
                except Exception as e:
                    print(f"[Nuggetizer]LLM Error at _score_nuggets: {str(e)}")
                    max_trails -= 1
                    if max_trails == 0:
                        print(f"[Nuggetizer]LLM Error at _score_nuggets: Failed to score nuggets after {max_trails} trails.")
                        scored_nuggets.extend([ScoredNugget(text=nugget.text, docids=nugget.docids, importance="failed") for nugget in batch_nuggets])
            start += window_size
        
        scored_nuggets = sorted(scored_nuggets, key=lambda x: (0 if x.importance == "vital" else 1, scored_nuggets.index(x)))
        return scored_nuggets

    def enrich_tree_with_nuggets(self, tree: QueryPassageTree, question: str) -> None:
        for seg_node in tree.iter_segment_nodes():
            nuggets: list[Nugget] = []
            try:
                nuggets = self._create_nuggets(question, seg_node.segment.text, self.config.max_nuggets_per_segment)
                print(f"[Nuggetizer]Created nuggets:\nquestion: {question}\nsegment: {seg_node.segment.text}\nnuggets: {nuggets}\n\n")
                for nugget in nuggets:
                    seg_node.nuggets.append(Nugget(text=nugget, docids=[seg_node.segment.docid]))
            except Exception as e:
                print(f"[Nuggetizer]LLM Error at enrich_tree_with_nuggets: {str(e)}")
        
        all_nuggets: list[Nugget] = [nugget for seg_node in tree.iter_segment_nodes() for nugget in seg_node.nuggets]
        merged_nuggets = self._merge_nuggets_by_embedding(question, all_nuggets) if len(all_nuggets) > 1 else all_nuggets
        scored_nuggets = self._score_nuggets(question, merged_nuggets)

        meta = {
            "num_raw_nuggets": len(all_nuggets),
            "num_merged_nuggets": len(merged_nuggets),
            "num_scored_nuggets": len(scored_nuggets),
            "num_vital_nuggets": len([nugget for nugget in scored_nuggets if nugget.importance == "vital"]),
            "num_okay_nuggets": len([nugget for nugget in scored_nuggets if nugget.importance == "okay"]),
            "num_failed_nuggets": len([nugget for nugget in scored_nuggets if nugget.importance == "failed"]),
        }

        return scored_nuggets, meta
    
    def analyze_search_complexity(self, tree: QueryPassageTree) -> Dict[str, Any]:        
        segment_node_with_nuggets = [seg_node for seg_node in tree.iter_segment_nodes() if len(seg_node.nuggets) > 0]
        depth_to_counts: Dict[int, Dict[str, int]] = {}
        for node in tree.iter_nodes():
            bucket = depth_to_counts.setdefault(node.depth, {"query_nodes": 0, "segment_nodes": 0, "total": 0})
            if isinstance(node, QueryNode):
                bucket["query_nodes"] += 1
            elif isinstance(node, SegmentNode) and node.nuggets:
                bucket["segment_nodes"] += 1
            bucket["total"] += 1
        
        max_depth = max((seg_node.depth for seg_node in segment_node_with_nuggets), default=0)

        nodes_per_depth: Dict[int, int] = {}
        for seg_node in segment_node_with_nuggets:
            nodes_per_depth[seg_node.depth] = nodes_per_depth.get(seg_node.depth, 0) + 1
        max_breadth = max(nodes_per_depth.values()) if nodes_per_depth else 0

        nuggets_per_depth: Dict[int, int] = {}
        for seg_node in segment_node_with_nuggets:
            nuggets_per_depth[seg_node.depth] = nuggets_per_depth.get(seg_node.depth, 0) + len(seg_node.nuggets)
        
        return {
            "max_depth": max_depth,
            "max_breadth": max_breadth,
            "nodes_per_depth": nodes_per_depth,
            "depth_distribution": depth_to_counts,
            "nuggets_per_depth": nuggets_per_depth,
            "num_nodes": len(tree.nodes),
            "num_query_nodes": len(tree.iter_query_nodes()),
            "num_segment_nodes": len(tree.iter_segment_nodes()),
            "num_segment_nodes_with_nuggets": len(segment_node_with_nuggets),
        }


def build_query_segment_tree(question, config) -> QueryPassageTree:
    tree_builder = TreeBuilder(config)
    return tree_builder.build(question)


def create_nuggets_from_tree(tree, question, config) -> None:
    nuggetizer = Nuggetizer(config)
    scored_nuggets, meta_1 = nuggetizer.enrich_tree_with_nuggets(tree, question)
    meta_2 = nuggetizer.analyze_search_complexity(tree)
    return scored_nuggets, {**meta_1, **meta_2}


def test_prompt(config):
    seg2q_client = LLM(config.seg2q_model, config.seg2q_model_api_url, config.seg2q_model_api_key)
    seg2q_judge_client = LLM(config.seg2q_judge_model, config.seg2q_judge_model_api_url, config.seg2q_judge_model_api_key)
    seg_judge_client = LLM(config.seg_judge_model, config.seg_judge_model_api_url, config.seg_judge_model_api_key)
    nuggetizer_client = LLM(config.nuggetizer_model, config.nuggetizer_model_api_url, config.nuggetizer_model_api_key)

    question = "Why did oil crash in 2015?"
    base_query = "reasons for 2015 oil price crash"
    passage = (
        "In late 2014, OPEC maintained output despite falling demand, leading to a global oil supply glut. "
        "U.S. shale production grew, and a strong dollar further pressured prices, culminating in 2015 declines."
    )

    if seg2q_client is not None:
        print("\n=== seg2q (rewrite) ===")
        messages = PromptTemplateManager.get_messages_for_seg2q(question, base_query, passage, max_num_new_queries=3)
        raw = seg2q_client.call(messages)
        print("Raw response:\n", raw)
        try:
            parsed = PromptTemplateManager.parse_seg2q_response(raw or "")
            print("Parsed queries:", parsed)
        except Exception as e:
            print("Parse error:", e)
    else:
        print("\n=== seg2q (rewrite) skipped ===")

    if seg2q_judge_client is not None:
        print("\n=== duplicate_judge ===")
        rewritten_query = "causes of the 2015 oil price crash"
        existing_queries = [
            "reasons for oil price collapse 2015",
            "2015 oil demand decline drivers",
            "OPEC production increase effect 2014 2015"
        ]
        messages = PromptTemplateManager.get_messages_for_duplicate_judge(rewritten_query, existing_queries)
        raw = seg2q_judge_client.call(messages, temperature=0.2)
        print("Raw response:\n", raw)
        try:
            parsed = PromptTemplateManager.parse_duplicate_judge_response(raw or "")
            print("Parsed is_duplicate:", parsed)
        except Exception as e:
            print("Parse error:", e)
    else:
        print("\n=== duplicate_judge skipped ===")

    if seg_judge_client is not None:
        print("\n=== time_judge ===")
        time_query = "In 2015, what factors caused the oil crash?"
        time_passage = (
            "A 2008 financial crisis triggered oil market turmoil. By 2015, persistent oversupply and OPEC policy kept prices low."
        )
        messages = PromptTemplateManager.get_messages_for_time_judge(time_query, time_passage)
        raw = seg_judge_client.call(messages, temperature=0.2)
        print("Raw response:\n", raw)
        try:
            parsed = PromptTemplateManager.parse_time_judge_response(raw or "")
            print("Parsed is_temporal_consistent:", parsed)
        except Exception as e:
            print("Parse error:", e)
    else:
        print("\n=== time_judge skipped ===")

    if nuggetizer_client is not None:
        print("\n=== create_nuggets ===")
        messages = PromptTemplateManager.get_messages_for_creating_nuggets(question, passage, creator_max_nuggets=3)
        raw = nuggetizer_client.call(messages)
        print("Raw response:\n", raw)
        try:
            parsed = PromptTemplateManager.parse_creating_nuggets_response(raw or "")
            print("Parsed nuggets:", parsed)
        except Exception as e:
            print("Parse error:", e)
    else:
        print("\n=== create_nuggets skipped ===")

    if nuggetizer_client is not None:
        print("\n=== merge_nuggets ===")
        sample_nuggets = [
            Nugget(text="OPEC kept output high in late 2014, creating an oversupply.", docids=["doc1"]),
            Nugget(text="High OPEC production in 2014 led to a global oil glut.", docids=["doc2"]),
            Nugget(text="A strong U.S. dollar put downward pressure on oil prices in 2015.", docids=["doc3"]),
        ]
        messages = PromptTemplateManager.get_messages_for_merging_nuggets(question, sample_nuggets)
        raw = nuggetizer_client.call(messages)
        print("Raw response:\n", raw)
        try:
            parsed = PromptTemplateManager.parse_merging_nuggets_response(raw or "")
            print("Parsed merged nuggets:", parsed)
        except Exception as e:
            print("Parse error:", e)
    else:
        print("\n=== merge_nuggets skipped ===")

    if nuggetizer_client is not None:
        print("\n=== score_nuggets ===")
        sample_nuggets_for_scoring = [
            Nugget(text="OPEC kept output high in late 2014, creating an oversupply.", docids=["doc1"]),
            Nugget(text="U.S. shale production growth increased supply.", docids=["doc4"]),
            Nugget(text="A strong U.S. dollar pressured oil prices in 2015.", docids=["doc3"]),
        ]
        messages = PromptTemplateManager.get_messages_for_scoring_nuggets(question, sample_nuggets_for_scoring)
        raw = nuggetizer_client.call(messages)
        print("Raw response:\n", raw)
        try:
            parsed = PromptTemplateManager.parse_scoring_nuggets_response(raw or "")
            print("Parsed labels:", parsed)
        except Exception as e:
            print("Parse error:", e)
    else:
        print("\n=== score_nuggets skipped ===")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set_path", type=str, required=True)
    parser.add_argument("--retriever_url", type=str, required=True)
    parser.add_argument("--embedder_model", type=str, required=True)

    parser.add_argument("--seg2q_model", type=str, required=True)
    parser.add_argument("--seg2q_model_api_url", type=str, required=True)
    parser.add_argument("--seg2q_model_api_key", type=str, required=True)
    parser.add_argument("--seg2q_judge_model", type=str, required=True)
    parser.add_argument("--seg2q_judge_model_api_url", type=str, required=True)
    parser.add_argument("--seg2q_judge_model_api_key", type=str, required=True)
    parser.add_argument("--seg_judge_model", type=str, required=True)
    parser.add_argument("--seg_judge_model_api_url", type=str, required=True)
    parser.add_argument("--seg_judge_model_api_key", type=str, required=True)
    parser.add_argument("--nuggetizer_model", type=str, required=True)
    parser.add_argument("--nuggetizer_model_api_url", type=str, required=True)
    parser.add_argument("--nuggetizer_model_api_key", type=str, required=True)

    parser.add_argument("--k_retrieved_passages", type=int, required=True)
    parser.add_argument("--hard_k_retrieved_passages", type=int, required=True)
    parser.add_argument("--threshold_score", type=float, default=None, required=False)
    parser.add_argument("--max_new_queries", type=int, required=True)
    parser.add_argument("--max_queries_per_passage", type=int, required=True)
    parser.add_argument("--max_nuggets_per_segment", type=int, required=True)
    parser.add_argument("--max_depth", type=int, required=True)
    parser.add_argument("--max_total_nodes", type=int, required=True)

    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--total_save_path", type=str, required=True)
    
    config = parser.parse_args()

    with open(config.train_set_path, "r") as file:
        data = [json.loads(line) for line in file]
    
    if os.path.exists(config.total_save_path):
        with open(config.total_save_path, "r") as file:
                existing_qids = [json.loads(line)["qid"] for line in file]
    else:
        existing_qids = []

    
    for item in tqdm(data, desc="Processing data"):
        if item["id"] in existing_qids:
            continue

        qid = item["id"]
        question = item["question"]

        intrinsic_scores = item["intrinsic_scores"]
        if intrinsic_scores.get("harmful", 0) > 0:
            continue
        if intrinsic_scores.get("incompleteness", 0) >= 8:
            continue
        if intrinsic_scores.get("assumptive", 0) >= 8:
            continue
        
        tree = build_query_segment_tree(question, config)
        scored_nuggets, metadata = create_nuggets_from_tree(tree, question, config)
        new_data = {
            "qid": qid,
            "query": question,
            "nuggets": [asdict(nugget) for nugget in scored_nuggets],
            "meta": metadata
        }

        with open(config.save_path, "a") as file:
            file.write(json.dumps(new_data) + "\n")

