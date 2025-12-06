import ast
import json
import numpy as np
import inspect
from llm_config import *
import google.generativeai as genai


class LLMRewardGenerator:
    def __init__(self, api_key, model, max_tokens, config_path):
        self.model = model
        self.max_tokens = max_tokens

        # 解析配置获取agent信息
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 提取agent名称和类型
        self.agent_names = []
        self.agent_types = []

        for supplier in config['components']['suppliers']:
            self.agent_names.append(supplier['name'])
            self.agent_types.append(supplier['type'])

        self.agent_names.append('battery')
        self.agent_types.append('battery')

        self.agent_names.append('customer')
        self.agent_types.append('customer')

        self.n_agents = len(self.agent_names)

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)

    def generate_reward_code(self, policy_description):
        """根据自然语言生成reward函数代码"""
        # 构建agent列表字符串
        agent_list = '\n'.join([
            f"  [{i}] {name} (type: {type_})"
            for i, (name, type_) in enumerate(zip(self.agent_names, self.agent_types))
        ])

        agent_names_str = '[' + ', '.join(self.agent_names) + ']'

        # 构建allocated_power字典的键列表（不包括customer，因为customer不在allocated_power中）
        agent_names_dict_keys = ', '.join([f"'{name}'" for name in self.agent_names[:-1]] + ["'main_grid'"])

        prompt = PROMPT_TEMPLATE.format(
            policy_description=policy_description,
            agent_list=agent_list,
            n_agents=self.n_agents,
            agent_names=agent_names_str,
            agent_names_dict_keys=agent_names_dict_keys
        )

        response = self.client.generate_content(prompt)
        return response.text

    def validate_and_compile(self, code):
        """验证并编译生成的代码"""
        code = code.replace("```python", "").replace("```", "").strip()

        # 安全检查
        try:
            tree = ast.parse(code)

            # 检查禁止的导入
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    if isinstance(node, ast.Import):
                        module_names = [n.name for n in node.names]
                    else:
                        module_names = [node.module] if node.module else []

                    if any(f in str(module_names) for f in FORBIDDEN_IMPORTS):
                        raise ValueError(f"Forbidden import detected: {module_names}")

            # 检查函数签名
            function_found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == REQUIRED_FUNCTION_NAME:
                    function_found = True

                    args = node.args
                    num_required = len(args.args) - len(args.defaults)

                    if num_required != NUM_REQUIRED_PARAMS:
                        raise ValueError(
                            f"Function must have exactly {NUM_REQUIRED_PARAMS} required parameters, got {num_required}"
                        )

                    if len(args.args) != NUM_TOTAL_PARAMS:
                        raise ValueError(
                            f"Function must have {NUM_TOTAL_PARAMS} total parameters, got {len(args.args)}"
                        )

                    actual_params = [arg.arg for arg in args.args]
                    if actual_params != REQUIRED_PARAMS:
                        raise ValueError(
                            f"Parameter names must be {REQUIRED_PARAMS}, got {actual_params}"
                        )

                    break

            if not function_found:
                raise ValueError(f"Function '{REQUIRED_FUNCTION_NAME}' not found in generated code")

        except SyntaxError as e:
            raise ValueError(f"Syntax error in generated code: {e}")

        # 构建测试数据
        test_auction_results = {
            'clearing_price': 0.25,
            'allocated_power': {name: 10.0 for name in self.agent_names[:-1]} | {'main_grid': 20.0},
            'actual_demand': 50.0,
            'curtailed_load': 5.0,
            'matched_trades': [
                {'buyer': 'customer', 'seller': self.agent_names[0], 'quantity': 10.0, 'price': 0.20}
            ]
        }

        test_env_data = {
            'wind_speed': 8.0,
            'solar_irradiance': 800.0,
            'base_load': 55.0,
            'time_hour': 12
        }

        # 测试调用
        namespace = {'np': np, 'numpy': np, 'Dict': dict}
        exec(code, namespace)

        if REQUIRED_FUNCTION_NAME not in namespace:
            raise ValueError(f"{REQUIRED_FUNCTION_NAME} not found in namespace after execution")
        reward_fn = namespace[REQUIRED_FUNCTION_NAME]

        try:
            result = reward_fn(0, test_auction_results, test_env_data, True)
        except Exception as e:
            raise ValueError(f"Function failed on test input: {e}")

        # 验证返回值
        expected_shape = (self.n_agents,)
        if not isinstance(result, np.ndarray):
            raise ValueError(f"Return type must be np.ndarray, got {type(result)}")

        if result.shape != expected_shape:
            raise ValueError(f"Return shape must be {expected_shape}, got {result.shape}")

        if not np.all(np.isfinite(result)):
            raise ValueError(f"Return values must be finite numbers, got {result}")

        return reward_fn