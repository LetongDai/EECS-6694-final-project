# Microgrid Controller with Transformer and LLM-generated Reward

Usage: 
```
python main.py [user_config path] [microgrid_config path]
```

## Introduction
The project implements a MADDPG for microgrid controller.
The agents learn optimal power generation policy from a real auction mechanism

Innovations:
- use MADDPG for continuous action prediction and better training result
- use transformers to improve the agents capability to learn relevance between observations
- integrate LLM-generated reward to allow automatic reward function generation that is suit for different demands


## Custom settings:
1. The user_config.json file contains:
    - LLM-generated reward settings
    - training settings
2. The microgrid_config.json file contains:
    - power generator settings. Allow adding or deleting power generators
    - battery and customer settings.
    - auction and other environment settings


## Notes:
The default setting enables the LLM generated reward.
To use this feature you need to provide a valid Google Gemini API key.
Otherwise you can disable the setting in user_config.json.