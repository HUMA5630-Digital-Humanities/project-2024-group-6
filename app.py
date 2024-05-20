import os
import sys
import threading
from itertools import chain

import anyio
import autogen
import gradio as gr
from autogen import Agent, AssistantAgent, OpenAIWrapper, UserProxyAgent, ConversableAgent
from autogen.code_utils import extract_code
from gradio import ChatInterface, Request
from gradio.helpers import special_args

LOG_LEVEL = "INFO"
TIMEOUT = 200


class myChatInterface(ChatInterface):
    async def _submit_fn(
        self,
        message: str,
        history_with_input: list[list[str | None]],
        request: Request,
        *args,
    ) -> tuple[list[list[str | None]], list[list[str | None]]]:
        history = history_with_input[:-1]
        inputs, _, _ = special_args(self.fn, inputs=[message, history, *args], request=request)

        if self.is_async:
            await self.fn(*inputs)
        else:
            await anyio.to_thread.run_sync(self.fn, *inputs, limiter=self.limiter)

        # history.append([message, response])
        return history, history


with gr.Blocks() as demo:

    def flatten_chain(list_of_lists):
        return list(chain.from_iterable(list_of_lists))

    class thread_with_trace(threading.Thread):
        # https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/
        # https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread
        def __init__(self, *args, **keywords):
            threading.Thread.__init__(self, *args, **keywords)
            self.killed = False
            self._return = None

        def start(self):
            self.__run_backup = self.run
            self.run = self.__run
            threading.Thread.start(self)

        def __run(self):
            sys.settrace(self.globaltrace)
            self.__run_backup()
            self.run = self.__run_backup

        def run(self):
            if self._target is not None:
                self._return = self._target(*self._args, **self._kwargs)

        def globaltrace(self, frame, event, arg):
            if event == "call":
                return self.localtrace
            else:
                return None

        def localtrace(self, frame, event, arg):
            if self.killed:
                if event == "line":
                    raise SystemExit()
            return self.localtrace

        def kill(self):
            self.killed = True

        def join(self, timeout=0):
            threading.Thread.join(self, timeout)
            return self._return

    def update_agent_history(recipient, messages, sender, config):
        if config is None:
            config = recipient
        if messages is None:
            messages = recipient._oai_messages[sender]
        message = messages[-1]
        message.get("content", "")
        # config.append(msg) if msg is not None else None  # config can be agent_history
        return False, None  # required to ensure the agent communication flow continues

    def _is_termination_msg(message):
        """Check if a message is a termination message.
        Terminate when no code block is detected. Currently only detect python code blocks.
        """
        if isinstance(message, dict):
            message = message.get("content")
            if message is None:
                return False
        cb = extract_code(message)
        contain_code = False
        for c in cb:
            # todo: support more languages
            if c[0] == "python":
                contain_code = True
                break
        return not contain_code

    def initialize_agents(config_list):
        
        assistant = AssistantAgent(
            name="assistant",
            max_consecutive_auto_reply=5,
            llm_config={
                # "seed": 42,
                "timeout": TIMEOUT,
                "config_list": config_list,
            },
        )

        angent_survey = ConversableAgent(
            name = "agent_survey",
            system_message="you are dedicated Learning Path Advisor. "
                    "the first task is understanding users goal and educational background, interests, career aspirations and other necessary information. "
                    "Interact with the user until you have sufficient information, or the user offers a message ending with 'Exit'."
                    "Please gently guide the user,ask questions one by one."
                    "based on the critic feedback, if needed, further ask user."
                    "summarizing the user's information but do not do recommandations.",
            llm_config={"config_list": config_list},
            is_termination_msg=lambda msg: "EXIT" in msg["content"],  # terminate 
            human_input_mode="NEVER",  # never ask for human input
        )
        angent_recommander =ConversableAgent(
            name = "angent_recommander",
            system_message="you are dedicated Learning Path Advisor."
                    "based on the user's information from 'agent_survey', critic and user, plan a learning path"
                    "your task is to help user navigate through the vast ocean of courses and specializations, finding the perfect path that aligns with user's background."
                    "the learning path should include necessary information of course such as course title, providers and thoughtful reasons for the recommendation"
                    "Then, ask the critic's opinion. and try to improve based on the opinion of critics"
                    "Rule 1. The total number of courses should be less than 4",
            llm_config={"config_list": config_list},
            is_termination_msg=lambda msg: "EXIT" in msg["content"],  
            human_input_mode="NEVER",  # never ask for human input
        )


        userproxy = UserProxyAgent(
            name="userproxy",
            system_message ="a human user",
            human_input_mode="NEVER",
            is_termination_msg=_is_termination_msg,
            max_consecutive_auto_reply=5,
            # code_execution_config=False,
            code_execution_config={
                #"last_n_messages": 2,
                "work_dir": "path_advisor",
                "use_docker": False,  # set to True or image name like "python:3" to use docker
            },
        )

        critic = AssistantAgent(
            name="Critic",
            system_message="Critic. Double check leanring path, reasons, from other agents and provide feedback. you should Reflect at least these questions"
                    "Q1: Whether the recommended course meets the user's interests or objective?"
                    "Q2: Do learning paths lead to higher motivation, or could they possibly lead to an overload of choices that paralyze some learners?"
                    "Q3: Is the content provided in-depth enough to foster a comprehensive understanding?"
                    "Q4: Is there a logical progression in the curriculum that builds on previous knowledge?"
                    "if you think the plan should imporved further, give the feedback to 'angent_recommander' and ask it to improve the learning path"
                    "if you think the plan is good enough, then ask the user if he would like to try the learning path?",
            llm_config={"config_list": config_list},
        )

        Learning_Path_summary = ConversableAgent(
            name = "Learning_Path_summary",
            system_message="You only followed by an approved leanring path plan by user." 
                    "Act as helpful and kind Learning Path Advisor, summarize the previous approved learning path for user, including course title, providers and thoughtful reasons for the recommendation"
                    "The tone should be informative, friendly, and supportive."
                    "And highlight the keypoints or keywords in orange"
                    "Your task is to provide a detailed summary of an approved leanring path plan to user. This overview is designed to give user clarity on the structure, objectives, and resources. the key elements are below:"
                    "Learning Path Overview: Goal Alignment; What were the initial goals  set at the start of this learning path? How do these objectives align with user's current background or personal development needs?"
                    "Curriculum Structure: provide a breakdown of the main topics and learning stages included in this path. What are the key outcomes expected at each stage, and how do they contribute to the overall goal?"
                    "Certification and Completion: Upon completing the learning path, what certificates or qualifications will be awarded? How do these credentials support further professional advancement or learning?"
                    "Future Learning Opportunities: What subsequent learning opportunities or advanced topics are recommended after completing this path?Are there any additional skills or areas of knowledge that suggest exploring to enhance professional growth?"
                    "Conclusion and give encouragement"
                    ,
            llm_config={"config_list": config_list},
            is_termination_msg=lambda msg: "EXIT" in msg["content"],  
            human_input_mode="NEVER",  # never ask for human input
        ) 
        #group chat with critic
        groupchat = autogen.GroupChat(agents=[userproxy, angent_survey, angent_recommander,critic,Learning_Path_summary], messages=[], max_round=20)
        manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

        # assistant.register_reply([Agent, None], update_agent_history)
        # userproxy.register_reply([Agent, None], update_agent_history)

        return userproxy, angent_survey, angent_recommander,critic,Learning_Path_summary, manager

    def chat_to_oai_message(chat_history):
        """Convert chat history to OpenAI message format."""
        messages = []
        if LOG_LEVEL == "DEBUG":
            print(f"chat_to_oai_message: {chat_history}")
        for msg in chat_history:
            messages.append(
                {
                    "content": msg[0].split()[0] if msg[0].startswith("exitcode") else msg[0],
                    "role": "user",
                }
            )
            messages.append({"content": msg[1], "role": "assistant"})
        return messages

    def oai_message_to_chat(oai_messages, sender):
        """Convert OpenAI message format to chat history."""
        chat_history = []
        messages = oai_messages[sender]
        if LOG_LEVEL == "DEBUG":
            print(f"oai_message_to_chat: {messages}")
        for i in range(0, len(messages), 2):
            chat_history.append(
                [
                    messages[i]["content"],
                    messages[i + 1]["content"] if i + 1 < len(messages) else "",
                ]
            )
        return chat_history

    def agent_history_to_chat(agent_history):
        """Convert agent history to chat history."""
        chat_history = []
        for i in range(0, len(agent_history), 2):
            chat_history.append(
                [
                    agent_history[i],
                    agent_history[i + 1] if i + 1 < len(agent_history) else None,
                ]
            )
        return chat_history

    def initiate_chat(config_list, user_message, chat_history):
        if LOG_LEVEL == "DEBUG":
            print(f"chat_history_init: {chat_history}")
        # agent_history = flatten_chain(chat_history)
        if len(config_list[0].get("api_key", "")) < 2:
            chat_history.append(
                [
                    user_message,
                    "Hi, nice to meet you!",
                ]
            )
            return chat_history
        else:
            llm_config = {
                # "seed": 42,
                "timeout": TIMEOUT,
                "config_list": config_list,
            }
            manager.llm_config.update(llm_config)
            manager.client = OpenAIWrapper(**manager.llm_config)

        manager.reset()
        oai_messages = chat_to_oai_message(chat_history)
        manager._oai_system_message_origin = manager._oai_system_message.copy()
        manager._oai_system_message += oai_messages

        try:
            userproxy.initiate_chat(manager, message=user_message)
            messages = userproxy.chat_messages
            chat_history += oai_message_to_chat(messages, manager)
            # agent_history = flatten_chain(chat_history)
        except Exception as e:
            # agent_history += [user_message, str(e)]
            # chat_history[:] = agent_history_to_chat(agent_history)
            chat_history.append([user_message, str(e)])

        manager._oai_system_message = manager._oai_system_message_origin.copy()
        if LOG_LEVEL == "DEBUG":
            print(f"chat_history: {chat_history}")
            # print(f"agent_history: {agent_history}")
        return chat_history

    def chatbot_reply_thread(input_text, chat_history, config_list):
        """Chat with the agent through terminal."""
        thread = thread_with_trace(target=initiate_chat, args=(config_list, input_text, chat_history))
        thread.start()
        try:
            messages = thread.join(timeout=TIMEOUT)
            if thread.is_alive():
                thread.kill()
                thread.join()
                messages = [
                    input_text,
                    "Timeout Error: Please check your API keys and try again later.",
                ]
        except Exception as e:
            messages = [
                [
                    input_text,
                    str(e) if len(str(e)) > 0 else "Invalid Request to OpenAI, please check your API keys.",
                ]
            ]
        return messages

    def chatbot_reply_plain(input_text, chat_history, config_list):
        """Chat with the agent through terminal."""
        try:
            messages = initiate_chat(config_list, input_text, chat_history)
        except Exception as e:
            messages = [
                [
                    input_text,
                    str(e) if len(str(e)) > 0 else "Invalid Request to OpenAI, please check your API keys.",
                ]
            ]
        return messages

    def chatbot_reply(input_text, chat_history, config_list):
        """Chat with the agent through terminal."""
        return chatbot_reply_thread(input_text, chat_history, config_list)

    def get_description_text():
        return """
        # Hello! 👋 My name is <span style="color:orange;">PathFinder</span>, 
        ## your dedicated Learning Path Advisor here.

        Welcome aboard!

        I am here to help you navigate through the vast ocean of courses and specializations, finding the perfect path that aligns with your career goals and educational interests.

        Whether you are looking to advance in your current field, pivot to a new industry, or simply explore new areas of knowledge, I'm here to guide you every step of the way!
        """

    def update_config():
        config_list = autogen.config_list_from_models(
            model_list=[os.environ.get("MODEL", "gpt-4")],
        )
        if not config_list:
            # set default openAI api here
            selected_model = ""
            selected_key = ""
            selected_url = ""

            config_list = [
                {
                    "api_key": selected_key,
                    "base_url": selected_url,
                    #"api_type": "azure",
                    #"api_version": "2023-07-01-preview",
                    "model": selected_model,
                }
            ]

        return config_list

    def set_params(model, oai_key, aoai_key, aoai_base):
        os.environ["MODEL"] = model
        os.environ["OPENAI_API_KEY"] = oai_key
        os.environ["AZURE_OPENAI_API_KEY"] = aoai_key
        os.environ["AZURE_OPENAI_API_BASE"] = aoai_base

    def respond(message, chat_history, model, oai_key, aoai_key, aoai_base):
        set_params(model, oai_key, aoai_key, aoai_base)
        config_list = update_config()
        chat_history[:] = chatbot_reply(message, chat_history, config_list)
        if LOG_LEVEL == "DEBUG":
            print(f"return chat_history: {chat_history}")
        return ""

    config_list= update_config()

    userproxy, angent_survey, angent_recommander,critic,Learning_Path_summary, manager = initialize_agents(config_list)

    description = gr.Markdown(get_description_text())

    with gr.Row() as params:
        txt_model = gr.Dropdown(
            label="Model",
            choices=[
                "gpt-4",
                "gpt-3.5-turbo",
            ],
            allow_custom_value=True,
            value="gpt-4",
            container=True,
        )
        txt_oai_key = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Enter OpenAI API Key",
            max_lines=1,
            show_label=True,
            container=True,
            type="password",
        )
        txt_aoai_key = gr.Textbox(
            label="Azure OpenAI API Key",
            placeholder="Enter Azure OpenAI API Key",
            max_lines=1,
            show_label=True,
            container=True,
            type="password",
        )
        txt_aoai_base_url = gr.Textbox(
            label="Base url",
            placeholder="Enter Base Url",
            max_lines=1,
            show_label=True,
            container=True,
            type="password",
        )

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(
            "user.png",
            (os.path.join(os.path.dirname(__file__), "advisor.png")),
        ),
        render=False,
        height=600,
    )

    txt_input = gr.Textbox(
        scale=4,
        show_label=False,
        placeholder="Enter text and press enter",
        container=False,
        render=False,
        autofocus=True,
    )

    chatiface = myChatInterface(
        respond,
        chatbot=chatbot,
        textbox=txt_input,
        additional_inputs=[
            txt_model,
            txt_oai_key,
            txt_aoai_key,
            txt_aoai_base_url,
        ],
        examples=[
            [" I am interested in data science but do not know where to start."],
            [" I'm a software developer and I want to learn about artificial intelligence. I have some experience with Python."],
            [" I have a background in literature and I'm interested in exploring more about the philosophical aspects of humanity. I would like to understand how philosophical theories have influenced human behavior and society."],
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
