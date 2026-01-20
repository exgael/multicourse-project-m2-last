from abc import ABC, abstractmethod
import prompts as prompts
from llm import llm_constructor
import os
import re

default_llm = llm_constructor('gpt4o')

def reset_token_counter():
        default_llm.reset_token_counter()

def get_token_counter():
        return default_llm.get_token_counter()

def set_model(model_name):
    global default_llm
    default_llm = llm_constructor(model_name)


class Agent(ABC):

    def __init__(self,description, llm = None):
        if llm is None:
            self.llm = default_llm
        else:
            self.llm = llm
        
        
        self.description =  description
        
    @abstractmethod
    def action(self,input_message):
        pass


class DefaultAgent(Agent):
    def __init__(self, llm=None):
          llm = llm_constructor('gpt4o', max_tokens=10000, temperature=1)
          description = "Generates things"
          super().__init__(description, llm)

    def action(self, prompt, input_message):
        message = input_message
        self.messages = [
            {
                "role": "system",
                "content": prompt
            }, 
            {
                "role": "user", 
                "content": message
            }
        ]
        
        resp = self.llm.inference(self.messages)
        
        return resp

class GraphSpecGenerator(Agent):
    def __init__(self, llm=None):
          description = "Generates ontology spec schema"
          super().__init__(description, llm)

    def action(self, input_message):
        prompt = prompts.schema_gen_instruct
        message = f"Generate the ontology graph for the following domain specification: {input_message}"
        self.messages = [
            {
                "role": "system",
                "content": prompt
            }, 
            {
                "role": "user", 
                "content": message
            }
        ]
        
        resp = self.llm.inference(self.messages)
        
        return resp


class GraphDepGenerator(Agent):
    def __init__(self, llm=None):
          description = "Generates ontology deps"
          super().__init__(description, llm)

    def action(self, input_message):
        prompt = prompts.dependency_gen
        message = f"Generate the dependencies of this ontology graph: {input_message}"
        self.messages = [
            {
                "role": "system",
                "content": prompt
            }, 
            {
                "role": "user", 
                "content": message
            }
        ]
        
        resp = self.llm.inference(self.messages)
        
        return resp
    

class DistGenerator(Agent):
    def __init__(self, llm=None):
          description = "Generates distributions"
          super().__init__(description, llm)

    def action(self, input_message):
        prompt = prompts.dist_spec_gen
        message = f"Generate the distribution of this attribute: {input_message}"
        self.messages = [
            {
                "role": "system",
                "content": prompt
            }, 
            {
                "role": "user", 
                "content": message
            }
        ]
        
        resp = self.llm.inference(self.messages)
        
        return resp

class StringListGenerator(Agent):
    def __init__(self, llm=None):
          llm = llm_constructor('gpt4o', max_tokens=10000, temperature=1)
          description = "Generates strings"
          super().__init__(description, llm)

    def action(self, input_message):
        prompt = prompts.string_list_gen
        message = f"Generate the list of values of this attribute that are very specific to the conditionning attributes and values (in context): {input_message}"
        self.messages = [
            {
                "role": "system",
                "content": prompt
            }, 
            {
                "role": "user", 
                "content": message
            }
        ]
        
        resp = self.llm.inference(self.messages)
        
        return resp
    
class ContextCardinalityGenerator(Agent):
    def __init__(self, llm=None):
          description = "Generates contextual cardinality"
          super().__init__(description, llm)

    def action(self, graphspec, contextual_keys):
        prompt = prompts.context_cardinality_gen
        message = f"Generate the contextual cardinality given this yaml ontology file and these dependencies: graphspec:\n ```yaml\n{graphspec}\n```\Contextual keys:\n```json{contextual_keys}\n```\n"
        self.messages = [
            {
                "role": "system",
                "content": prompt
            }, 
            {
                "role": "user", 
                "content": message
            }
        ]
        
        resp = self.llm.inference(self.messages)
        
        return resp

class AgentCategorySBS(Agent):
    def __init__(self, llm=None):
        description = ""
        super().__init__(description, llm)
        self.prompts = {
            "task_event": prompts.TASK_EVENT_SBS,
            #"parallel_insert":prompts.PARALLEL_EXISTING_SBS,
            #"parallel_append":prompts.PARALLEL_APPEND_SBS,
            "gateway_insert":prompts.GENERAL_GATEWAY_INSERT,
            "gateway_append": prompts.GENERAL_GATEWAY_APPEND
        }



    def action(self, process_sentence, graph_json, last_nodes, category):
        
        prompt = self.prompts[category]

        template = prompts.SBS_REASONING
        
        template = template.format(process_sentence = process_sentence, graph_json = graph_json, last_nodes = last_nodes)
        
        
        self.messages = [
            {
                "role": "system",
                "content": prompt
            }, 
            {
                "role": "user", 
                "content": template
            }
        ]

        resp = self.llm.inference(self.messages)
        
        return resp

    def extend_messages(self, new_messages):
        self.messages.extend(new_messages)

    def extend_and_action(self, new_messages):
        self.messages.extend(new_messages)
        resp = self.llm.inference(self.messages)
        return resp




class AgentSplitValidator(Agent):
    def __init__(self, llm=None):
        description = ""
        super().__init__(description, llm)
    
    def action(self, process_step):
        prompt = prompts.SPLIT_VALIDATIOR
        template = """
Now analyze the following step: {process_step}.
Return the json step between ```json ``` tags. 
```json
...
```
""".format(process_step = process_step)
        
        messages = [
            {
                "role": "system",
                "content": prompt
            }, 
            {
                "role": "user", 
                "content": template
            }
        ]

        resp = self.llm.inference(messages)
        
        return resp


