# Group Project for HUMA5630 Digital Humanities

| Student Name | UID | UST email | Github username |
| ------------ | --- | --------- | --------------- |
|     Xu,Kaiyan| 20923894    |    kxuav@connect.ust.hk       |    [@KieranXu](https://github.com/KieranXu)             |


# Topic
This project is focused on developing an course recommender AI agent called PathFinder, powered by AutoGen. This AI equips PathFinder with the ability to help users navigate through Coursera's vast array of courses and specializations. It provides personalized recommendations based on the user's educational background, interests, and career goals. 

# Agent group

**Survey Agent**: This agent gathers essential information about the user's educational background, interests, and career aspirations to understand their goals better.

**Recommend Agent**: After determining the user's needs from the survey agent, this agent designs a personalized learning pathway. It provides course information and reasons why these courses are suitable for the user.

**Critic**: The critic agent summarizes the information, reflects on it, and checks the facts. It provides feedback to ensure the learning pathway is accurate and tailored to the user's needs.


# How to use
## set default OpenAI api in app.py (optional)
```
selected_model = ""
selected_key = ""
selected_url = ""
```
## Run app
```
# Install dependencies
pip install -U -r requirements.txt

# Launch app
python app.py
```

# preview
![preview](https://github.com/HUMA5630-Digital-Humanities/project-2024-group-6/assets/128702515/7a572151-964c-4a20-aa90-f777dc9e23cd)

# Note

There are still some bugs that have not been resolved.
1. Some messages sent to the group manager by the AI agent will be presented in the chat list as the user's role.
    Possible reason: The order and quantity of dialogues in the chatgroup are uncertain
    Possible solution: Mark the source of each message in the chat history, differentiate it from the user agent
    
2. When the group chat carries on multiple rounds of dialogue, a token may exceed the agent's limit, causing an error.
3. There is still a problem with the illusion, such as the wrong course url provided.
